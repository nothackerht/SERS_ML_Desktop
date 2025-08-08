import os
import numpy as np
import torch
from collections import Counter
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

import modules.bayes_visualizations as bvvis
from modules.preprocessing import Preprocessing
from modules.bayes_visualizations import (
    plot_parity, plot_convergence_curve, plot_evaluations_scatter,
    plot_hyperparam_heatmap, plot_feature_importance, plot_residuals,
    plot_shap_summary
)
from modules.bayes_opt import optimize_xgb_with_cv


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

    fold_records = []
    hyper_list = []

    # ---------- PHASE 1 — Ensemble Hyperparameters (Per-Fold) ----------
    for i in range(n_ex):
        # Keep outputs in the chain directory directly (no double-nesting)
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
        hyper_list.append(best_hp)

        # Plots for BayesOpt diagnostics in this fold directory
        plot_convergence_curve(res)
        plot_evaluations_scatter(res)
        plot_hyperparam_heatmap(res, 'max_depth', 'learning_rate')

        # Train final fold model on preprocessed X_pool with fold-optimal HPs
        mdl_final = XGBRegressor(
            **dict(zip(names, best_hp)),
            tree_method='hist',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_state=42
        )
        mdl_final.fit(X_proc, y_pool)
        joblib.dump(mdl_final, os.path.join(fold_dir, 'mdl_final.pkl'))

        # Fold diagnostics
        plot_feature_importance(mdl_final, top_n=20)
        plot_residuals(mdl_final, X_proc, y_pool)
        plot_shap_summary(mdl_final, X_proc)

        # Predict held-out (transform-only path)
        preds = mdl_final.predict(Xh)
        avg_pred = float(np.mean(preds))
        avg_rmse = float(np.sqrt(mean_squared_error([y_hold], [avg_pred])))

        # Parity (fold-level)
        plot_parity(f"Fold_{i}_{'+'.join(chain)}", [y_hold], [avg_pred])

        fold_records.append({
            'fold': i,
            'held_out_sample': sample_ids_ex[i],
            'preproc': '+'.join(chain),
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

    global_records = []

    for i in range(n_ex):
        # Optional separate directory for global phase outputs
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
            **dict(zip(names, mode_hp)),
            tree_method='hist',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_state=42
        )
        mdl_g.fit(X_train_final, y_pool)
        joblib.dump(mdl_g, os.path.join(global_dir, 'mdl_global.pkl'))

        preds = mdl_g.predict(X_hold_final)
        avg_pred = float(np.mean(preds))
        avg_rmse = float(np.sqrt(mean_squared_error([y_hold], [avg_pred])))

        # Parity (global phase, per fold)
        plot_parity(f"Global_Fold_{i}_{'+'.join(chain)}", [y_hold], [avg_pred])

        # (Optional) global diagnostics
        plot_feature_importance(mdl_g, top_n=20)
        plot_residuals(mdl_g, X_train_final, y_pool)
        plot_shap_summary(mdl_g, X_train_final)

        global_records.append({
            'fold': i,
            'held_out_sample': sample_ids_ex[i],
            'preproc': '+'.join(chain),
            'mode_hp': mode_hp,
            'global_pred_mean': avg_pred,
            'global_pred_std': float(np.std(preds)),
            'global_rmse': avg_rmse,
            'y_test_spectra': np.repeat(y_hold, len(preds)),
            'y_pred_spectra': preds.tolist()
        })

    return fold_records, global_records
