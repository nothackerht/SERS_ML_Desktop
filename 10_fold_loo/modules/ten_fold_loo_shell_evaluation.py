# -*- coding: utf-8 -*-
"""
10_fold_loo_shell_evaluation.py

Leave-One-Out evaluation on a blinded test set using 10-fold GroupKFold,
with 5-fold inner CV for hyperparameter tuning (only hyperparams, fixed preprocessing).
Runs each model over each preprocessing chain separately, avoiding data leakage.
Saves per-model, per-prep, per-target .csv and parity plots (with error bars, hyperparams & prep).
"""
# --- add right after the docstring, before importing numpy/pandas ---
import os
# cap nested threading to avoid MKL/OMP explosions on Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (save-to-file only), safe with joblib


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

# NEW: compute RMSE/R2 on sample-means (9 spectra per sample)
from sklearn.metrics import mean_squared_error, r2_score
def _rmse_r2_on_sample_means(y_true_vec, y_pred_vec, reps=9):
    y_true_vec = np.asarray(y_true_vec).reshape(-1)
    y_pred_vec = np.asarray(y_pred_vec).reshape(-1)
    assert y_true_vec.size == y_pred_vec.size
    if y_true_vec.size % reps != 0:
        raise ValueError("Vector length must be a multiple of reps")
    s_true = y_true_vec.reshape(-1, reps).mean(axis=1)
    s_pred = y_pred_vec.reshape(-1, reps).mean(axis=1)
    rmse = float(np.sqrt(mean_squared_error(s_true, s_pred)))
    r2   = float(r2_score(s_true, s_pred))
    return rmse, r2

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
    include_controls: bool = False,
    control_labels: tuple[str, ...] = ("Control",),  # ← accept many
):

    """
    Run 10-fold LOO over the test set. Uses the hardened data_loader to align spectra↔sample IDs.
    If include_controls=True, Controls are loaded but rows with non-finite targets are dropped automatically.
    """
    os.makedirs(output_dir, exist_ok=True)

    include_types = ("DM1",) + (control_labels if include_controls else ())
    print(f"[LOO] include_controls={include_controls}, control_labels={control_labels}")
    print(f"[LOO] include_types={include_types}")


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
    print(f"[LOO] Loaded TRAIN rows: {len(meta_train)}; TEST rows: {len(meta_test)}")
    print("[LOO] TRAIN types:", meta_train['Type'].value_counts(dropna=False).to_dict())
    print("[LOO] TEST  types:", meta_test['Type'].value_counts(dropna=False).to_dict())

    # ---- 2) Keep only samples with finite targets (handles Controls gracefully) ----
    mask_train = build_finite_mask(meta_train, target_column)
    mask_test  = build_finite_mask(meta_test,  target_column)

    # Filter spectra (9 columns per sample) and metadata
    all_train = all_train[:, _cols_from_mask(mask_train)]
    all_test  = all_test[:,  _cols_from_mask(mask_test)]
    meta_train = meta_train.loc[mask_train].reset_index(drop=True)
    meta_test  = meta_test.loc[mask_test].reset_index(drop=True)
    print(f"[LOO] target_column='{target_column}'")
    print(f"[LOO] TRAIN finite mask kept {mask_train.sum()}/{mask_train.size} rows")
    print(f"[LOO] TEST  finite mask kept {mask_test.sum()}/{mask_test.size} rows")
    if 'Type' in meta_train.columns:
        print("[LOO] TRAIN kept by type:", meta_train.loc[mask_train, 'Type'].value_counts().to_dict())
    if 'Type' in meta_test.columns:
        print("[LOO] TEST  kept by type:", meta_test.loc[mask_test, 'Type'].value_counts().to_dict())

    # Consistency checks (9 spectra per sample)
    n_train = meta_train.shape[0]
    n_test  = meta_test.shape[0]
    assert all_train.shape[1] == 9 * n_train, "train spectra mismatch"
    assert all_test.shape[1]  == 9 * n_test,  "test spectra mismatch"
    


  
    # <<< INSERT HERE >>>
    results = []            # you already had this
    detail_rows = []        # NEW: per-fold, per-hyperparam metrics go here
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
                        n_intervals=100,
                        n_components=2,
                        cv_folds=5,
                        threshold=0.3
                    )
                else:
                    sel = np.arange(X_raw.shape[1])

                # ---- 5) Inner CV for hyperparameter tuning (parallelized) ----
                def _evaluate_hp(hp):
                    fold_rmses = []   # <-- was: fold_rmses, fold_r2s = []
                    fold_r2s   = []   # initialize the second list separately
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
                
                        rmse, r2 = _rmse_r2_on_sample_means(y_vl, preds, reps=9)
                        fold_rmses.append(rmse)
                        fold_r2s.append(r2)
                
                    return float(np.mean(fold_rmses)), float(np.mean(fold_r2s)), hp

                
                # Safer on Windows: fewer workers, use threads, and avoid nested BLAS threads
                max_workers = min(4, os.cpu_count() or 1)
                hp_results = Parallel(n_jobs=max_workers, prefer="threads", verbose=10)(
                    delayed(_evaluate_hp)(hp) for hp in hyperparam_grids.get(model_name, [])
                )
                if not hp_results:
                    raise ValueError(f"No hyperparameters provided for model '{model_name}'.")
                
                # choose best by CV RMSE
                best_idx  = int(np.argmin([r[0] for r in hp_results]))
                best_rmse, best_r2, best_hp = hp_results[best_idx]

                # ---- 6) Retrain & evaluate EACH hyperparameter on full pool + held-out ----
                
                # 6.1 unsupervised preprocessing on full pool (train + test_except_held)
                X_full_raw = X_raw.T
                prep_full = Preprocessing(X_full_raw)
                Xp_full_unsup = prep_full.preprocess(other_methods)          # (n_features, n_spectra_total)
                
                # 6.2 slice to selected intervals and transpose to samples×features
                Xp_full_sel = Xp_full_unsup[sel, :]                          # (n_selected_features, n_spectra_total)
                Xp_full     = Xp_full_sel.T.astype(np.float32)               # (n_spectra_total, n_selected_features)
                
                # 6.3 targets for full pool
                y_full_input = y_vals.reshape(-1, 1).astype(np.float32) if model_name == 'sipls' else y_vals.astype(np.float32)
                
                # 6.4 transform held-out block the same way
                loo_raw   = all_test[:, held_cols]                           # (n_features, 9)
                prep_loo  = Preprocessing(loo_raw)
                Xp_loo_unsup = prep_loo.preprocess(other_methods)            # (n_features, 9)
                Xp_loo_sel   = Xp_loo_unsup[sel, :]                          # (n_selected_features, 9)
                X_loo32      = Xp_loo_sel.T.astype(np.float32)               # (9, n_selected_features)
                
                # 6.5 final sanity checks
                _assert_finite("Xp_full", Xp_full)
                _assert_finite("X_loo32", X_loo32)
                _assert_finite("y_full_input", y_full_input)
                
                # For every hyperparameter candidate:
                for cv_rmse, cv_r2, hp in hp_results:
                    mdl = get_model_by_name(model_name, **hp)
                    mdl.fit(Xp_full, y_full_input)
                
                    # re-train score (on the full pool) – report on sample means
                    preds_full = mdl.predict(Xp_full)
                    preds_full = preds_full.ravel() if hasattr(preds_full, "ndim") and preds_full.ndim > 1 else np.ravel(preds_full)
                    retr_rmse, retr_r2 = _rmse_r2_on_sample_means(y_vals, preds_full, reps=9)
                
                    # final prediction for this outer fold, this hp
                    raw_preds = mdl.predict(X_loo32)
                    raw_preds = raw_preds.ravel() if hasattr(raw_preds, "ndim") and raw_preds.ndim > 1 else np.ravel(raw_preds)
                    test_mean = float(raw_preds.mean())
                    test_true = float(meta_test[target_column].values[held_sample_idx])
                
                    # store per-fold, per-hp
                    detail_rows.append({
                        'fold': fold,
                        'model': model_name,
                        'preprocess': '+'.join(prep_chain),
                        'hyperparams': json.dumps(hp),
                        'is_best_by_cv_rmse': bool(hp == best_hp),
                        'cv_rmse': cv_rmse,
                        'cv_r2':   cv_r2,
                        'retrain_rmse': retr_rmse,
                        'retrain_r2':   retr_r2,
                        'test_pred_mean': test_mean,
                        'test_true':      test_true
                    })

    # ---- 7) Save detailed per-fold table ----
    df = pd.DataFrame(detail_rows)
    df.to_csv(os.path.join(output_dir, 'loo_per_fold_all_hyperparams.csv'), index=False)
    
    # ---- 8) Aggregate by (model, preprocess, hyperparams) across all outer folds ----
    def _final_metrics(group):
        # inner-CV and retrain: mean over folds
        cv_rmse_mean     = group['cv_rmse'].mean()
        cv_r2_mean       = group['cv_r2'].mean()
        retrain_rmse_mean= group['retrain_rmse'].mean()
        retrain_r2_mean  = group['retrain_r2'].mean()
        # final test performance: computed from held-out preds across folds
        final_rmse = float(np.sqrt(mean_squared_error(group['test_true'].values,
                                                      group['test_pred_mean'].values)))
        final_r2   = float(r2_score(group['test_true'].values,
                                    group['test_pred_mean'].values))
        return pd.Series({
            'cv_rmse_mean': cv_rmse_mean,
            'cv_r2_mean':   cv_r2_mean,
            'retrain_rmse_mean': retrain_rmse_mean,
            'retrain_r2_mean':   retrain_r2_mean,
            'final_rmse': final_rmse,
            'final_r2':   final_r2
        })
    
    by_combo = (
        df.groupby(['model','preprocess','hyperparams'], as_index=False)
          .apply(_final_metrics)
          .reset_index(drop=True)
    )
    
    # also keep a compact “best-by-CV” view (optional)
    best_by_cv = (
        by_combo.sort_values(['model','preprocess','cv_rmse_mean'])
                .groupby(['model','preprocess'], as_index=False)
                .first()
    )
    
    # ---- 9) Save to Excel (multiple sheets) ----
    xlsx_path = os.path.join(output_dir, f"loo_report_{target_column}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        df.to_excel(writer,      sheet_name='per_fold', index=False)
        by_combo.to_excel(writer, sheet_name='by_combo', index=False)
        best_by_cv.to_excel(writer, sheet_name='best_by_cv', index=False)
    
    print(f"[LOO] Wrote detailed Excel report → {xlsx_path}")
    
    # (optional) keep a light CSV with final metrics only
    by_combo.to_csv(os.path.join(output_dir, 'loo_metrics_by_combo.csv'), index=False)
    
    return df, by_combo




