# -*- coding: utf-8 -*-
"""
10_fold_loo_shell_evaluation.py

Leave-One-Out evaluation on a blinded test set using 10-fold GroupKFold,
with 5-fold inner CV for hyperparameter tuning (only hyperparams, fixed preprocessing).
Runs each model over each preprocessing chain separately, avoiding data leakage.
Saves per-model, per-prep, per-target .csv and parity plots (with error bars, hyperparams & prep).
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed

from modules.interval_shell import get_model_by_name
from modules.data_loader import load_data, build_finite_mask
from modules.preprocessing import Preprocessing


# ---------- helpers ----------
def _sample_cols(sample_indices, reps=9):
    # expand sample indices → spectrum column indices
    idx = np.asarray(sample_indices, dtype=int)
    return np.concatenate([np.arange(i*reps, (i+1)*reps) for i in idx]) if idx.size else np.array([], dtype=int)

def _assert_finite(name, arr):
    if not np.isfinite(arr).all():
        bad = int(np.size(arr) - np.isfinite(arr).sum())
        raise ValueError(f"Non-finite values in {name} (count={bad}).")

def _cols_from_mask(mask, reps=9):
    idx = np.where(mask)[0]
    return _sample_cols(idx, reps=reps)


# ---------- main API ----------
def leave_one_out_test_evaluation(
    train_data_dir: str,
    train_meta_path: str,
    test_data_dir: str,
    test_meta_path: str,
    model_list: list[str],
    hyperparam_grids: dict[str, list[dict]],
    preprocess_grid: list[list[str]],
    output_dir: str,
    target_column: str = "target_SI",
    include_controls: bool = False,   # NEW: simple switch
):
    """
    Run 10-fold LOO over the test set. Uses the hardened data_loader to align spectra↔sample IDs.
    If include_controls=True, Controls are loaded but rows with non-finite targets are dropped automatically.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1) Load aligned datasets (order-independent, ID-based) ----
    include_types = ("DM1", "Control") if include_controls else ("DM1",)

    # Training
    _, _, all_train, meta_train = load_data(
        train_data_dir,
        metadata_path=train_meta_path,
        include_types=include_types,
        return_filenames=False,
        # strict=True,
        strict=False,
        report_samples=5,      # prints a small alignment report
    )

    # Testing
    _, _, all_test, meta_test = load_data(
        test_data_dir,
        metadata_path=test_meta_path,
        include_types=include_types,
        return_filenames=False,
        # strict=True,
        strict=False,
        report_samples=5,
    )

    # ---- 2) Keep only samples with finite targets (handles Controls gracefully) ----
    mask_train = build_finite_mask(meta_train, target_column)
    mask_test  = build_finite_mask(meta_test,  target_column)

    # Filter spectra (9 columns per sample) and metadata
    all_train = all_train[:, _cols_from_mask(mask_train)]
    all_test  = all_test[:,  _cols_from_mask(mask_test)]
    meta_train = meta_train.loc[mask_train].reset_index(drop=True)
    meta_test  = meta_test.loc[mask_test].reset_index(drop=True)

    # Consistency checks (9 spectra per sample)
    n_train = meta_train.shape[0]
    n_test  = meta_test.shape[0]
    assert all_train.shape[1] == 9 * n_train, "train spectra mismatch"
    assert all_test.shape[1]  == 9 * n_test,  "test spectra mismatch"

    # Group vector for test spectra (9 per sample)
    test_groups = np.repeat(np.arange(n_test), 9)

    results = []

    # ---- 3) Outer LOO per test sample ----
    for model_name in model_list:
        for prep_chain in preprocess_grid:

            # Precompute list of "other" (unsupervised) methods; IntervalPLS is handled specially
            other_methods = [m for m in prep_chain if m != 'IntervalPLS']

            for fold in range(n_test):
                # Which test sample is held out this fold?
                held_sample_idx = fold
                held_cols = np.arange(held_sample_idx * 9, (held_sample_idx + 1) * 9)

                # Columns for all *other* test samples
                ti_sample_idx = np.setdiff1d(np.arange(n_test), [held_sample_idx])
                ti_cols = _sample_cols(ti_sample_idx)

                # ---- Build combined pool (train + test_except_held) ----
                # X_raw: rows = spectra, cols = wavenumbers
                X_raw = np.hstack([all_train, all_test[:, ti_cols]]).T
                # y_vals: one target per spectrum (repeat each sample target 9×)
                y_vals = np.hstack([
                    np.repeat(meta_train[target_column].values, 9),
                    np.repeat(meta_test[target_column].values[ti_sample_idx], 9)
                ]).astype(np.float32)

                # One group id per sample (for GroupKFold), repeated 9×
                groups = np.concatenate([
                    np.repeat(np.arange(n_train), 9),
                    np.repeat(np.arange(len(ti_sample_idx)) + n_train, 9)
                ])

                # ---- 4) OUTER IntervalPLS feature selection (once per outer fold) ----
                if 'IntervalPLS' in prep_chain:
                    X_outer = X_raw
                    y_outer = y_vals
                    sel = Preprocessing(X_outer).select_intervals_by_pls_r2(
                        X_outer, y_outer,
                        n_intervals=150,
                        n_components=2,
                        cv_folds=5,
                        threshold=0.22
                    )
                else:
                    sel = np.arange(X_raw.shape[1])

                # ---- 5) Inner CV for hyperparameter tuning (parallelized) ----
                def _evaluate_hp(hp):
                    fold_rmses = []
                    inner = GroupKFold(n_splits=5)
                    for iti, ito in inner.split(X_raw, y_vals, groups=groups):
                        X_tr_raw = X_raw[iti].T
                        X_vl_raw = X_raw[ito].T
                        y_tr, y_vl = y_vals[iti], y_vals[ito]

                        # apply only unsupervised steps
                        Xp_tr = Preprocessing(X_tr_raw).preprocess(other_methods).T.astype(np.float32)
                        Xp_vl = Preprocessing(X_vl_raw).preprocess(other_methods).T.astype(np.float32)

                        # interval selection slice
                        Xp_tr, Xp_vl = Xp_tr[:, sel], Xp_vl[:, sel]

                        # get model and fit
                        mdl = get_model_by_name(model_name, **hp)
                        y_tr_fit = y_tr.reshape(-1, 1) if model_name == 'sipls' else y_tr
                        mdl.fit(Xp_tr, y_tr_fit.astype(np.float32))

                        preds = mdl.predict(Xp_vl)
                        preds = preds.ravel() if hasattr(preds, "ndim") and preds.ndim > 1 else np.ravel(preds)

                        # score on sample means
                        s_true = y_vl.reshape(-1, 9).mean(axis=1)
                        s_pred = preds.reshape(-1, 9).mean(axis=1)
                        fold_rmses.append(np.sqrt(mean_squared_error(s_true, s_pred)))

                    return np.mean(fold_rmses), hp

                hp_results = Parallel(n_jobs=-1, verbose=10)(
                    delayed(_evaluate_hp)(hp)
                    for hp in hyperparam_grids.get(model_name, [])
                )
                if not hp_results:
                    raise ValueError(f"No hyperparameters provided for model '{model_name}'.")
                best_rmse, best_hp = min(hp_results, key=lambda x: x[0])

                # ---- 6) Retrain on full pool & predict the held-out 9 spectra ----
                # 6.1 unsupervised preprocessing on full pool
                X_full_raw = X_raw.T
                prep_full = Preprocessing(X_full_raw)
                Xp_full_unsup = prep_full.preprocess(other_methods)          # (n_features, n_spectra_total)

                # 6.2 slice to selected intervals and transpose to samples×features
                Xp_full_sel = Xp_full_unsup[sel, :]                          # (n_selected_features, n_spectra_total)
                Xp_full = Xp_full_sel.T.astype(np.float32)                   # (n_spectra_total, n_selected_features)

                # 6.3 target vector for full pool
                y_full_input = y_vals.reshape(-1, 1).astype(np.float32) if model_name == 'sipls' else y_vals.astype(np.float32)

                # 6.4 transform the held-out block likewise
                loo_raw = all_test[:, held_cols]                             # (n_features, 9)
                prep_loo = Preprocessing(loo_raw)
                Xp_loo_unsup = prep_loo.preprocess(other_methods)            # (n_features, 9)
                Xp_loo_sel = Xp_loo_unsup[sel, :]                            # (n_selected_features, 9)
                X_loo32 = Xp_loo_sel.T.astype(np.float32)                    # (9, n_selected_features)

                # 6.5 final sanity (helps catch any NaN propagation)
                _assert_finite("X_full32(final)", Xp_full)
                _assert_finite("X_loo32(final)",  X_loo32)
                _assert_finite("y_full_input(final)", y_full_input)

                # 6.6 fit + predict
                mdl = get_model_by_name(model_name, **best_hp)
                mdl.fit(Xp_full, y_full_input)
                raw_preds = mdl.predict(X_loo32)
                raw_preds = raw_preds.ravel() if hasattr(raw_preds, "ndim") and raw_preds.ndim > 1 else np.ravel(raw_preds)

                # 6.7 store mean/std of the nine predictions
                pred_mean = raw_preds.mean()
                pred_std  = raw_preds.std()

                results.append({
                    'fold':       fold,
                    'model':      model_name,
                    'preprocess': '+'.join(prep_chain),
                    'hyperparams': best_hp,
                    'held_out':   int(held_sample_idx),
                    'predictions': raw_preds.tolist(),
                    'pred_mean':  float(pred_mean),
                    'pred_std':   float(pred_std),
                    'true':       float(meta_test[target_column].values[held_sample_idx])
                })

    # ---- 7) Save predictions & metrics ----
    df = pd.DataFrame(results)
    df['hyperparams'] = df['hyperparams'].apply(json.dumps)
    df.to_csv(os.path.join(output_dir, 'loo_predictions.csv'), index=False)

    metrics = df.groupby(['model','preprocess']).apply(
        lambda g: pd.Series({
            'RMSE': np.sqrt(mean_squared_error(g['true'], g['pred_mean'])),
            'R2':   r2_score(g['true'], g['pred_mean'])
        })
    )
    metrics.to_csv(os.path.join(output_dir, 'loo_metrics.csv'))

    # ---- 8) Plot parity for the best preproc per model ----
    best_per_model = (
        metrics
        .reset_index()
        .sort_values(['model','RMSE'], ascending=[True,True])
        .drop_duplicates('model', keep='first')
    )

    for _, row in best_per_model.iterrows():
        m, prep, rmse, r2 = row['model'], row['preprocess'], row['RMSE'], row['R2']
        grp = df[(df['model'] == m) & (df['preprocess'] == prep)]

        y_true = grp['true'].values
        y_pred = grp['pred_mean'].values

        plt.figure(figsize=(6,6))
        plt.errorbar(y_true, y_pred, yerr=grp['pred_std'], fmt='o', capsize=4)
        mn, mx = y_true.min(), y_true.max()
        plt.plot([mn,mx], [mn,mx], 'r--')

        plt.xlabel(target_column, fontweight='bold')
        plt.ylabel(target_column, fontweight='bold')
        plt.title(
            f"{m}\n"
            f"Preproc = {prep or 'None'}\n"
            f"Best Hyperparams: {grp['hyperparams'].mode().iloc[0]}\n"
            f"RMSE = {rmse:.3f},  R² = {r2:.3f}",
            fontsize=10,
            loc='center'
        )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"parity_{m.replace(' ','_')}.png"), dpi=300)
        plt.show()

    return df, metrics


# ---------- example CLI ----------
if __name__ == "__main__":
    # Example invocation
    train_data_dir     = r"C:\Users\notha\Downloads\Data\data"
    train_meta_path    = r"C:\Users\notha\Downloads\Data\y_metadata.csv"
    test_data_dir      = r"C:\Users\notha\Downloads\Data\data_test_updated"
    test_meta_path     = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata_test_updated_in_order.csv"
    results_root       = r"C:\Users\notha\Downloads\Data\Results"

    model_list = ['sipls']
    hyperparam_grids = {
        'sipls': [
            {'n_components': 2, 'device': 'cpu'},
            # add more if you like
        ]
    }
    preprocess_grid = [[]]  # try others as needed
    targets = [("Splicing Index", "target_SI")]

    for nice_name, col in targets:
        out_dir = os.path.join(results_root, f"10_fold_LOO_{col}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n\n### Running LOO for {nice_name} ({col}) → {out_dir}")
        loo_df, loo_metrics = leave_one_out_test_evaluation(
            train_data_dir   = train_data_dir,
            train_meta_path  = train_meta_path,
            test_data_dir    = test_data_dir,
            test_meta_path   = test_meta_path,
            model_list       = model_list,
            hyperparam_grids = hyperparam_grids,
            preprocess_grid  = preprocess_grid,
            output_dir       = out_dir,
            target_column    = col,
            include_controls = False,   # set True to include Controls with finite targets
        )
        print(f"--- {nice_name} head ---\n", loo_df.head())
        print(f"--- {nice_name} metrics ---\n", loo_metrics)
