import os
import numpy as np
import torch
from collections import Counter
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
from datetime import datetime

import modules.bayes_visualizations as bvvis
from modules.preprocessing import Preprocessing
from modules.bayes_visualizations import (
    plot_parity, plot_convergence_curve, plot_evaluations_scatter,
    plot_hyperparam_heatmap, plot_feature_importance, plot_residuals,
    plot_shap_summary
)
from modules.bayes_opt import optimize_xgb_with_cv


# ───────────────────────────── Helper: save a portable model bundle ─────────────────────────────

def _save_model_bundle(save_path, *, preprocessor, model, params, meta=None):
    """
    Save a portable bundle: fitted preprocessor + fitted model + params + meta.
    Also saves the Booster JSON next to it for version-robust portability.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bundle = {
        "preprocessor": preprocessor,        # fitted Preprocessing instance
        "model": model,                      # fitted XGBRegressor
        "params": params,                    # dict of hyperparameters
        "meta": {**(meta or {}), "timestamp": datetime.now().isoformat(timespec="seconds")},
    }
    joblib.dump(bundle, save_path, compress=3)

    # Booster JSON (robust across xgboost versions)
    try:
        booster_json = os.path.splitext(save_path)[0] + ".json"
        model.get_booster().save_model(booster_json)
    except Exception:
        # If booster not available (or model not fitted), skip silently
        pass


def run_outer_loo(raw_tr, y_tr_meta, raw_ex, y_ex_meta, sample_ids_ex, chain, xgb_space, base_out, n_calls=25):
    """
    Two-phase LOO:
      Phase 1 (Ensemble): per-fold BayesOpt on X_pool (train + 10/11 test); fit outer preprocessing on X_pool,
                          transform held-out only; train with fold-optimal HPs; predict held-out.
      Phase 2 (Global):   aggregate fold-optimal HPs across 11 folds into a global set (binned mode);
                          re-run outer training per fold with the fixed global HPs.
    """
    os.makedirs(base_out, exist_ok=True)

    n_ex = raw_ex.shape[1] // 9
    y_tr = np.repeat(y_tr_meta, 9)
    y_ex = np.repeat(y_ex_meta, 9)
    ext_idx = np.repeat(np.arange(n_ex), 9)
    names = [d.name for d in xgb_space]
    chain_name = '+'.join(chain)
    target_name = "target_SI"  # adjust if you run different targets per call

    fold_records = []
    hyper_list = []

    # For writing small collection files at the end
    ensemble_bundle_paths = []
    global_bundle_paths = []

    # ---------- PHASE 1 — Ensemble Hyperparameters (Per-Fold) ----------
    for i in range(n_ex):
        fold_dir = os.path.join(base_out, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)

        # Route all plot outputs for this fold here
        bvvis.OUTPUT_DIR = fold_dir

        # Hold-out split
        hold_mask = (ext_idx == i)
        X_hold = raw_ex[:, hold_mask]
        y_hold = y_ex_meta[i]

        # Pool = train + remaining 10 test samples
        rem_mask = ~hold_mask
        X_rem = raw_ex[:, rem_mask]
        X_pool = np.hstack([raw_tr, X_rem])
        y_pool = np.concatenate([y_tr, y_ex[rem_mask]])

        # Sample-level groups (9 spectra/sample)
        grp_tr = np.repeat(np.arange(len(y_tr_meta)), 9)
        grp_ex = len(y_tr_meta) + ext_idx[rem_mask]
        groups = np.concatenate([grp_tr, grp_ex])

        # Outer preprocessing: fit on X_pool, transform held-out only
        prep = Preprocessing(ipls_threshold=0.2)
        X_proc = prep.fit_transform(X_pool, chain, y=y_pool).T
        Xh = prep.transform(X_hold, chain).T

        # Inner BayesOpt with GroupKFold(5) and fold-safe preprocessing
        res = optimize_xgb_with_cv(
            X_pool, y_pool, groups, chain, xgb_space, y_tr_meta, y_ex_meta, n_calls=n_calls
        )

        # Record best hyperparameters for this fold
        best_hp = tuple(res.x)
        best_params = dict(zip(names, best_hp))
        hyper_list.append(best_hp)

        # BayesOpt diagnostics
        plot_convergence_curve(res)
        plot_evaluations_scatter(res)
        plot_hyperparam_heatmap(res, 'max_depth', 'learning_rate')

        # Train final fold model on preprocessed X_pool with fold-optimal HPs
        mdl_final = XGBRegressor(
            **best_params,
            tree_method='hist',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric='rmse',
        )
        mdl_final.fit(X_proc, y_pool)

        # Save ensemble model bundle (preprocessor + model + params + meta)
        ens_dir = os.path.join(fold_dir, 'models')
        ens_path = os.path.join(ens_dir, f'ensemble_fold{i:02d}.joblib')
        _save_model_bundle(
            ens_path,
            preprocessor=prep,
            model=mdl_final,
            params=best_params,
            meta={
                "type": "ensemble",
                "chain": chain_name,
                "target": target_name,
                "fold": i,
                "held_out": str(sample_ids_ex[i]),
                "train_size_samples": int(X_proc.shape[0]),
                "train_size_spectra": int(X_proc.shape[0]),  # sample-level orientation used in training
            }
        )
        ensemble_bundle_paths.append(ens_path)

        # Fold diagnostics
        plot_feature_importance(mdl_final, top_n=20)
        plot_residuals(mdl_final, X_proc, y_pool)
        plot_shap_summary(mdl_final, X_proc)

        # Predict held-out (transform-only path)
        preds = mdl_final.predict(Xh)
        avg_pred = float(np.mean(preds))
        avg_rmse = float(np.sqrt(mean_squared_error([y_hold], [avg_pred])))

        # Parity (fold-level)
        plot_parity(f"Fold_{i}_{chain_name}", [y_hold], [avg_pred])

        fold_records.append({
            'fold': i,
            'held_out_sample': sample_ids_ex[i],
            'preproc': chain_name,
            'best_hp': best_hp,
            'intervals': None,
            'fold_pred_mean': avg_pred,
            'fold_pred_std': float(np.std(preds)),
            'fold_rmse': avg_rmse,
            'y_test_spectra': np.repeat(y_hold, len(preds)),
            'y_pred_spectra': preds.tolist()
        })

    # ---------- PHASE 2 — Global Hyperparameters (Mode Across Folds) ----------
    # Quantize floats before taking mode, so near-identical values group together
    def _bin_hp_tuple(hp_tuple):
        binned = []
        for v in hp_tuple:
            binned.append(round(v, 3) if isinstance(v, float) else v)
        return tuple(binned)

    binned = [_bin_hp_tuple(hp) for hp in hyper_list]
    mode_hp = Counter(binned).most_common(1)[0][0]
    mode_params = dict(zip(names, mode_hp))

    global_records = []

    for i in range(n_ex):
        global_dir = os.path.join(base_out, "global", f"fold_{i}")
        os.makedirs(global_dir, exist_ok=True)
        bvvis.OUTPUT_DIR = global_dir

        # Hold-out split
        hold_mask = (ext_idx == i)
        X_hold = raw_ex[:, hold_mask]
        y_hold = y_ex_meta[i]

        # Pool = train + remaining 10 test samples
        rem_mask = ~hold_mask
        X_rem = raw_ex[:, rem_mask]
        X_pool = np.hstack([raw_tr, X_rem])
        y_pool = np.concatenate([y_tr, y_ex[rem_mask]])

        # Outer preprocessing (global phase): fit on X_pool, transform held-out only
        prep = Preprocessing(ipls_threshold=0.2)
        X_train_final = prep.fit_transform(X_pool, chain, y=y_pool).T
        X_hold_final = prep.transform(X_hold, chain).T

        # Train with fixed global HPs
        mdl_g = XGBRegressor(
            **mode_params,
            tree_method='hist',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric='rmse',
        )
        mdl_g.fit(X_train_final, y_pool)

        # Save global model bundle (preprocessor + model + params + meta)
        glo_dir = os.path.join(global_dir, 'models')
        glo_path = os.path.join(glo_dir, f'global_modehp_fold{i:02d}.joblib')
        _save_model_bundle(
            glo_path,
            preprocessor=prep,
            model=mdl_g,
            params=mode_params,
            meta={
                "type": "global",
                "chain": chain_name,
                "target": target_name,
                "fold": i,
                "held_out": str(sample_ids_ex[i]),
                "train_size_samples": int(X_train_final.shape[0]),
                "train_size_spectra": int(X_train_final.shape[0]),
            }
        )
        global_bundle_paths.append(glo_path)

        preds = mdl_g.predict(X_hold_final)
        avg_pred = float(np.mean(preds))
        avg_rmse = float(np.sqrt(mean_squared_error([y_hold], [avg_pred])))

        # Parity (global phase, per fold)
        plot_parity(f"Global_Fold_{i}_{chain_name}", [y_hold], [avg_pred])

        # (Optional) global diagnostics
        plot_feature_importance(mdl_g, top_n=20)
        plot_residuals(mdl_g, X_train_final, y_pool)
        plot_shap_summary(mdl_g, X_train_final)

        global_records.append({
            'fold': i,
            'held_out_sample': sample_ids_ex[i],
            'preproc': chain_name,
            'mode_hp': mode_hp,
            'global_pred_mean': avg_pred,
            'global_pred_std': float(np.std(preds)),
            'global_rmse': avg_rmse,
            'y_test_spectra': np.repeat(y_hold, len(preds)),
            'y_pred_spectra': preds.tolist()
        })

    # ---------- Small collection files (paths to all per-fold bundles) ----------
    models_root = os.path.join(base_out, "models")
    os.makedirs(models_root, exist_ok=True)

    # List of ensemble bundle files for this chain
    joblib.dump(
        {
            "type": "ensemble_collection",
            "chain": chain_name,
            "target": target_name,
            "fold_bundle_paths": ensemble_bundle_paths
        },
        os.path.join(models_root, f"ensemble_all_{chain_name}.joblib"),
        compress=3
    )

    # List of global bundle files for this chain
    joblib.dump(
        {
            "type": "global_collection",
            "chain": chain_name,
            "target": target_name,
            "fold_bundle_paths": global_bundle_paths,
            "mode_params": mode_params
        },
        os.path.join(models_root, f"global_all_{chain_name}.joblib"),
        compress=3
    )

    return fold_records, global_records
