# -*- coding: utf-8 -*-
"""
10_fold_loo_shell_evaluation.py

Leave-One-Out evaluation on a blinded test set using 10-fold GroupKFold,
with 5-fold inner CV for hyperparameter tuning (only hyperparams, fixed preprocessing).
Runs each model over each preprocessing chain separately, avoiding data leakage.
Saves per-model, per-prep, per-target .csv and parity plots (with error bars, hyperparams & prep).
"""
from itertools import product
# --- add right after the docstring, before importing numpy/pandas ---
import os
# cap nested threading to avoid MKL/OMP explosions on Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (save-to-file only), safe with joblib

import math
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

# --- performance knobs (put near the top, after imports) ---
import tempfile, psutil, gc

LOGICAL_CORES = os.cpu_count() or 32
PHYSICAL_CORES = getattr(psutil, "cpu_count", lambda logical=False: None)(logical=False) or max(1, LOGICAL_CORES // 2)

# Use ~75% of logical cores for CPU models; small pool for GPU XGB
WORKERS_CPU = min(24, max(4, LOGICAL_CORES - LOGICAL_CORES // 4))   # e.g. 32 -> 24
WORKERS_XGB = 2


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
# ---------- model cache helpers ----------
import hashlib, joblib
from datetime import datetime

def _canonicalize_params(d: dict) -> dict:
    """Sort keys and normalize types for stable hashing."""
    import numpy as _np
    def canon(x):
        if isinstance(x, dict):
            return {k: canon(x[k]) for k in sorted(x)}
        if isinstance(x, (list, tuple)):
            return [canon(v) for v in x]
        if isinstance(x, _np.ndarray):
            return x.tolist()
        return x
    return canon(d)

def _split_fingerprint(train_sample_tags) -> str:
    """
    Stable fingerprint for the *outer-fold* training set.
    `train_sample_tags` should be a list like [('tr', 0), ... , ('te', 3)].
    """
    s = "|".join(f"{a}:{b}" for a, b in sorted(train_sample_tags))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def _sel_signature(sel_indices: np.ndarray) -> str:
    """Compact signature of the selected feature indices (iPLS result)."""
    h = hashlib.md5(sel_indices.astype(np.int64).tobytes()).hexdigest()
    return h[:12]

def _cache_key(*, model_name: str, params: dict, preprocess: list[str],
               target_col: str, train_fp: str, sel_sig: str) -> str:
    payload = {
        "model": model_name,
        "params": _canonicalize_params(params),
        "preprocess": list(preprocess or []),
        "target": target_col,
        "train_fp": train_fp,
        "sel_sig": sel_sig,
        "sklearn": __import__("sklearn").__version__,
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _bundle_path(cache_dir: str, key: str, suffix: str = "joblib") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{key}.{suffix}")

def save_model_bundle(cache_dir: str, key: str, *, preprocessor, model, meta: dict | None = None):
    bundle = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "preprocessor": preprocessor,
        "model": model,
        "meta": meta or {},
    }
    joblib.dump(bundle, _bundle_path(cache_dir, key), compress=3)

    # If XGBoost, also persist Booster to JSON for cross-version robustness
    try:
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            booster.save_model(_bundle_path(cache_dir, key + "_booster", "json"))
    except Exception:
        pass

def load_model_bundle_if_exists(cache_dir: str, key: str):
    p = _bundle_path(cache_dir, key)
    if os.path.exists(p):
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None

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
    control_labels: tuple[str, ...] = ("Control",),
    pure_nesting: bool = True,   # <<< NEW: True = refit transforms inside inner CV (pure)

):

    """
    Run 10-fold LOO over the test set. Uses the hardened data_loader to align spectra↔sample IDs.
    If include_controls=True, Controls are loaded but rows with non-finite targets are dropped automatically.
    """
    # Set up output + caches
    os.makedirs(output_dir, exist_ok=True)
    print(f"[LOO] pure_nesting={pure_nesting}  (True=pipeline-style, refit transforms inside inner CV)")

    # model cache lives under this LOO output folder
    CACHE_DIR = os.path.join(output_dir, "_model_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # joblib memmap folder lives under the same output_dir
    JOBLIB_TEMP_DIR = os.path.join(output_dir, "_joblib_tmp")
    os.makedirs(JOBLIB_TEMP_DIR, exist_ok=True)
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", JOBLIB_TEMP_DIR)

    include_types = ("DM1",) + (control_labels if include_controls else ())
    print(f"[LOO] include_controls={include_controls}, control_labels={control_labels}")
    print(f"[LOO] include_types={include_types}")


    # Training
    _, _, all_train, filenames_train, meta_train = load_data(
        train_data_dir,
        metadata_path=train_meta_path,
        include_types=include_types,
        return_filenames=True,
        strict=False,
        report_samples=5,
        mapping_csv_path=os.path.join(output_dir, "sample_docs_train.csv")
    )


    # Testing
    _, _, all_test, filenames_test, meta_test = load_data(
        test_data_dir,
        metadata_path=test_meta_path,
        include_types=include_types,
        return_filenames=True,
        strict=False,
        report_samples=5,
        mapping_csv_path=os.path.join(output_dir, "sample_docs_test.csv")
    )
    
    print(f"[LOO] Loaded TRAIN rows: {len(meta_train)}; TEST rows: {len(meta_test)}")
    print("[LOO] TRAIN types:", meta_train['Type'].value_counts(dropna=False).to_dict())
    print("[LOO] TEST  types:", meta_test['Type'].value_counts(dropna=False).to_dict())
    # ---------- provenance helpers & logger (computed once per run) ----------
    spectral_checks_log = []  # rows to write at the end
    
    def _id_from_filename(fn: str) -> str:
        p = fn.split("_")
        if len(p) < 2:
            raise ValueError(f"Bad filename for ID extraction: {fn}")
        return f"{p[0]}_{p[1]}"
    
    def _provenance_result(filenames_per_col, mask, meta, reps=9):
        """
        Verify that, AFTER masking (dropping non-finite targets), the kept 9*samples
        filenames map 1:1 to meta['FilePrefix'] repeated 9×, in order.
        Also assert the 9-wide contiguous block structure.
        """
        cols_keep = _cols_from_mask(mask, reps=reps)  # col indices to keep in all_* arrays
        ids_per_col = np.array([_id_from_filename(f) for f in filenames_per_col])
        kept_ids = ids_per_col[cols_keep]                              # length = 9 * kept_samples
        expected_ids = np.repeat(meta["FilePrefix"].values, reps)      # same length
    
        ok_align = bool(np.array_equal(kept_ids, expected_ids))
        msg_align = "filenames↔meta order OK (post-mask)" if ok_align else "MISMATCH in filenames↔meta (post-mask)"
    
        # 9-wide contiguous blocks after masking (guaranteed by our slicing; still record it)
        n_samples = meta.shape[0]
        ok_blocks = (kept_ids.reshape(n_samples, reps)[:, 0] == meta["FilePrefix"].values).all()
        msg_blocks = "9-wide contiguous blocks confirmed" if ok_blocks else "Block contiguity failed"
    
        return {
            "ok_align": ok_align,
            "msg_align": msg_align,
            "ok_blocks": ok_blocks,
            "msg_blocks": msg_blocks,
        }
    
    def _log_prov_row(model, preprocess, target_col, split, check_name, passed, message):
        spectral_checks_log.append({
            "model": model,
            "preprocess": "+".join(preprocess) if isinstance(preprocess, (list, tuple)) else str(preprocess),
            "target_column": target_col,
            "split": split,        # "train" or "test"
            "check": check_name,   # e.g. "align_ids" / "contiguous_blocks"
            "passed": bool(passed),
            "message": message,
        })

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
        print("[LOO] TRAIN kept by type:", meta_train['Type'].value_counts().to_dict())
    if 'Type' in meta_test.columns:
        print("[LOO] TEST  kept by type:",  meta_test['Type'].value_counts().to_dict())
        
    # -------- compute provenance checks ONCE (train & test) --------
    train_prov = _provenance_result(filenames_train, mask_train, meta_train, reps=9)
    test_prov  = _provenance_result(filenames_test,  mask_test,  meta_test,  reps=9)
    
    # Hard fail early if something is off (so you don't waste cycles)
    if not (train_prov["ok_align"] and train_prov["ok_blocks"] and test_prov["ok_align"] and test_prov["ok_blocks"]):
        raise AssertionError(
            f"[PROVENANCE] Failed checks — "
            f"train_align={train_prov['ok_align']}, train_blocks={train_prov['ok_blocks']}, "
            f"test_align={test_prov['ok_align']}, test_blocks={test_prov['ok_blocks']}"
        )


    # Consistency checks (9 spectra per sample)
    n_train = meta_train.shape[0]
    n_test  = meta_test.shape[0]
    assert all_train.shape[1] == 9 * n_train, "train spectra mismatch"
    assert all_test.shape[1]  == 9 * n_test,  "test spectra mismatch"
    
    # --- iPLS hyperparam grid (tuned inside inner CV if 'IntervalPLS' is in the chain) ---
    ipls_param_grid = [
        # threshold mode
        dict(n_intervals=30,  n_components=2, select_mode="threshold", threshold=0.2, top_k=None),
        dict(n_intervals=50,  n_components=2, select_mode="threshold", threshold=0.2, top_k=None),
        dict(n_intervals=75,  n_components=2, select_mode="threshold", threshold=0.3, top_k=None),
        # top-k mode
        dict(n_intervals=75,  n_components=2, select_mode="topk",      threshold=0.0, top_k=10),
        dict(n_intervals=100, n_components=2, select_mode="topk",      threshold=0.0, top_k=15),
        dict(n_intervals=150, n_components=2, select_mode="topk",      threshold=0.0, top_k=20),
    ]
    # split stability repeats: robust outer, cheap inner
    STAB_REPEATS_OUTER = 5   # used in _get_outer_sel (outer fold selection)
    STAB_REPEATS_INNER = 1   # used inside _evaluate_combo (inner CV scoring)
    STAB_KEEP          = 0.6


  
    # <<< INSERT HERE >>>
    results = []            # you already had this
    detail_rows = []        # NEW: per-fold, per-hyperparam metrics go here
    
    # ---- 3) Outer LOO per test sample ----
    for model_name in model_list:
        for prep_chain in preprocess_grid:
    
            # Precompute list of "other" (unsupervised) methods; IntervalPLS is handled specially
            other_methods = [m for m in prep_chain if m != 'IntervalPLS']
    
            # --- log that this (model, preprocess) combo inherits the global provenance checks
            _log_prov_row(model_name, prep_chain, target_column, "train", "align_ids",
                          train_prov["ok_align"], train_prov["msg_align"])
            _log_prov_row(model_name, prep_chain, target_column, "train", "contiguous_blocks",
                          train_prov["ok_blocks"], train_prov["msg_blocks"])
    
            _log_prov_row(model_name, prep_chain, target_column, "test", "align_ids",
                          test_prov["ok_align"], test_prov["msg_align"])
            _log_prov_row(model_name, prep_chain, target_column, "test", "contiguous_blocks",
                          test_prov["ok_blocks"], test_prov["msg_blocks"])
            for fold in range(n_test):
                # Which test sample is held out this fold?
                held_sample_idx = fold
                held_cols = np.arange(held_sample_idx * 9, (held_sample_idx + 1) * 9)

                # Columns for all *other* test samples
                ti_sample_idx = np.setdiff1d(np.arange(n_test), [held_sample_idx])
                ti_cols = _sample_cols(ti_sample_idx)
                # Build a stable tag list for this fold's training samples:
                #  - all training samples (tag 'tr', indices 0..n_train-1)
                #  - all test samples EXCEPT the held one (tag 'te', using ti_sample_idx)
                train_sample_tags = [('tr', int(i)) for i in range(n_train)] + [('te', int(i)) for i in ti_sample_idx]
                train_fp = _split_fingerprint(train_sample_tags)  # e.g., 'a1b2c3d4e5f6'

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

 
                          
                # ---- 4) OUTER: build unsupervised-preprocessed matrix (no interval slicing here) ----
                prep_sel = Preprocessing()
                prep_sel.fit(X_raw.T, other_methods)                         # (n_features, n_spectra_total)
                X_outer_unsup = prep_sel.transform(X_raw.T, other_methods).T # (n_spectra_total, n_features)
                
                # cache for final (outer) selection per iPLS setting in this fold
                ipls_outer_cache = {}  # key = tuple(sorted(ipls_hp.items())) -> (sel_final, n_intervals_selected, n_features_kept)
                
                def _get_outer_sel(ipls_hp):
                    """Compute (and cache) the outer-fold iPLS selection for a given ipls_hp dict."""
                    if ipls_hp is None:  # no IntervalPLS in the chain
                        sel = np.arange(X_outer_unsup.shape[1], dtype=int)
                        return sel, np.nan, sel.size

                    key = tuple(sorted(ipls_hp.items()))
                    if key in ipls_outer_cache:
                        return ipls_outer_cache[key]

                    sel = Preprocessing().select_intervals_grouped(
                        X_outer_unsup, y_vals, groups,
                        plot=False,
                        stability_repeats=STAB_REPEATS_OUTER,
                        stability_keep=STAB_KEEP,
                        **ipls_hp
                    )

                    n_features = X_outer_unsup.shape[1]
                    blocks = np.array_split(np.arange(n_features, dtype=int), ipls_hp["n_intervals"])
                    n_kept = sum(np.intersect1d(b, sel).size > 0 for b in blocks)
                    ipls_outer_cache[key] = (sel, n_kept, int(len(sel)))
                    return ipls_outer_cache[key]

                # shared across all (model_hp, ipls_hp) jobs in THIS outer fold
                # key: (ipls_key, tuple(sorted(unique_groups_tr))) -> sel_inner
                ipls_inner_cache = {}


    
                # ---- 5) Inner CV for hyperparameter tuning (parallelized) ----
                def _evaluate_combo(model_hp, ipls_hp):
                    fold_rmses, fold_r2s = [], []
                    # be safe if unique groups < 5
                    n_inner = max(2, min(5, np.unique(groups).size))
                    inner = GroupKFold(n_splits=n_inner)

                

                
                    for iti, ito in inner.split(X_outer_unsup, y_vals, groups=groups):
                        y_tr, y_vl = y_vals[iti], y_vals[ito]
                        groups_tr, groups_vl = groups[iti], groups[ito]
                    
                        if pure_nesting:
                            # Refit ALL unsupervised transforms on the inner-training spectra only
                            prep_in = Preprocessing()
                            prep_in.fit(X_raw[iti].T, other_methods)                     # spectra: (n_features, n_train_spectra)
                            X_tr_unsup = prep_in.transform(X_raw[iti].T, other_methods).T  # -> (n_train_spectra, n_features')
                            X_vl_unsup = prep_in.transform(X_raw[ito].T, other_methods).T  # -> (n_val_spectra,   n_features')
                        else:
                            # Not-pure: reuse the outer-fit transforms
                            X_tr_unsup, X_vl_unsup = X_outer_unsup[iti], X_outer_unsup[ito]

                
                        # compute inner selection on training-only (always leakage-free)
                        if 'IntervalPLS' in prep_chain and ipls_hp is not None:
                            if pure_nesting:
                                # No caching: transform basis changes with each inner training set
                                sel_inner = Preprocessing().select_intervals_grouped(
                                    X_tr_unsup, y_tr, groups_tr,
                                    plot=False,
                                    stability_repeats=STAB_REPEATS_INNER,
                                    stability_keep=STAB_KEEP,
                                    **ipls_hp
                                )
                            else:
                                # Reuse cache keyed by (ipls settings, unique groups in training) — OK because transform is fixed
                                ipls_key  = tuple(sorted(ipls_hp.items()))
                                gsig      = tuple(np.unique(groups_tr))
                                cache_key = (ipls_key, gsig)
                                sel_inner = ipls_inner_cache.get(cache_key)
                                if sel_inner is None:
                                    sel_inner = Preprocessing().select_intervals_grouped(
                                        X_tr_unsup, y_tr, groups_tr,
                                        plot=False,
                                        stability_repeats=STAB_REPEATS_INNER,
                                        stability_keep=STAB_KEEP,
                                        **ipls_hp
                                    )
                                    ipls_inner_cache[cache_key] = sel_inner
                        
                            Xp_tr = X_tr_unsup[:, sel_inner]
                            Xp_vl = X_vl_unsup[:, sel_inner]
                        else:
                            Xp_tr, Xp_vl = X_tr_unsup, X_vl_unsup


                
                        # model fit/eval
                        mdl = get_model_by_name(model_name, **model_hp)
                        y_tr_fit = y_tr.reshape(-1, 1) if model_name == 'sipls' else y_tr
                        mdl.fit(Xp_tr.astype(np.float32), y_tr_fit.astype(np.float32))
                        preds = mdl.predict(Xp_vl)
                        preds = preds.ravel() if getattr(preds, "ndim", 1) > 1 else np.ravel(preds)
                
                        rmse, r2 = _rmse_r2_on_sample_means(y_vl, preds, reps=9)
                        fold_rmses.append(rmse)
                        fold_r2s.append(r2)
                
                    # <-- return AFTER finishing all inner folds
                    se = 0.0 if len(fold_rmses) <= 1 else float(np.std(fold_rmses, ddof=1) / math.sqrt(len(fold_rmses)))
                    return (
                        float(np.mean(fold_rmses)),
                        se,
                        float(np.mean(fold_r2s)),
                        model_hp,
                        ipls_hp,
                    )

                
                # ---- 5) INNER: parallel over model HP × iPLS HP (if iPLS is in the chain) ----
                if 'IntervalPLS' in prep_chain:
                    jobs = product(hyperparam_grids.get(model_name, []), ipls_param_grid)
                else:
                    jobs = ((hp, None) for hp in hyperparam_grids.get(model_name, []))
                
                # Resolve joblib temp folder under this run's output_dir
                joblib_tmp = globals().get("JOBLIB_TEMP_DIR") or os.path.join(output_dir, "_joblib_tmp")
                os.makedirs(joblib_tmp, exist_ok=True)
                
                # Choose workers based on model type (constants defined near imports)
                _total = LOGICAL_CORES
                max_workers = WORKERS_XGB if model_name == "xgboost" else WORKERS_CPU
                
                # Processes + memmap for big arrays -> faster & lower RAM duplication
                hp_results = Parallel(
                    n_jobs=max_workers,
                    backend="loky",
                    verbose=10,
                    max_nbytes="256M",          # memmap arrays larger than 256MB
                    temp_folder=joblib_tmp,
                    mmap_mode="r",
                )(
                    delayed(_evaluate_combo)(hp, ipls_hp) for hp, ipls_hp in jobs
                )
                
                if not hp_results:
                    raise ValueError(f"No hyperparameters provided for model '{model_name}'.")
                
                # hp_results tuples: (cv_rmse, cv_rmse_se, cv_r2, model_hp, ipls_hp)
                means = np.array([r[0] for r in hp_results], dtype=float)
                ses   = np.array([r[1] for r in hp_results], dtype=float)
                
                best_mean_idx = int(np.argmin(means))
                one_se_threshold = means[best_mean_idx] + ses[best_mean_idx]
                
                # candidates within one SE of the minimum
                candidates = [i for i, m in enumerate(means) if m <= one_se_threshold]
                
                # tie-breaker: prefer the *simplest* iPLS (fewest features; then fewer intervals)
                def _complexity(idx):
                    ipls_hp = hp_results[idx][4]  # r[4] is ipls_hp
                    sel, n_intervals_kept, n_feats_kept = _get_outer_sel(ipls_hp)
                    # smaller is better
                    return (n_feats_kept, 0 if np.isnan(n_intervals_kept) else int(n_intervals_kept))
                
                best_idx = min(candidates, key=_complexity)
                
                best_cv_rmse, best_cv_rmse_se, best_cv_r2, best_model_hp, best_ipls_hp = hp_results[best_idx]
                
                # Compute outer selection ONCE for the best iPLS setting
                sel_best, n_intervals_selected_best, n_features_kept_best = _get_outer_sel(best_ipls_hp)
                if best_ipls_hp is not None:
                    print(f"[LOO] fold={fold} | best iPLS={best_ipls_hp} | "
                          f"kept {n_intervals_selected_best} intervals / {n_features_kept_best} features")

                
                # ---- 6) Retrain & evaluate on full pool + held-out ----
                # 6.1 unsupervised preprocessing on full pool (train + test_except_held)
                X_full_raw = X_raw.T  # (n_features, n_spectra_total)
                prep_full = Preprocessing()
                prep_full.fit(X_full_raw, other_methods)
                Xp_full_unsup = prep_full.transform(X_full_raw, other_methods)   # (n_features, n_spectra_total)
                
                # 6.3 targets for full pool   <-- keep this
                y_full_input = y_vals.reshape(-1, 1).astype(np.float32) if model_name == 'sipls' else y_vals.astype(np.float32)
                
                # transform held-out block with *unsupervised* steps only
                loo_raw = all_test[:, held_cols]                   # (n_features, 9)
                Xp_loo_unsup = prep_full.transform(loo_raw, other_methods)  # (n_features, 9)
                
                _assert_finite("Xp_full_unsup", Xp_full_unsup)
                _assert_finite("Xp_loo_unsup",  Xp_loo_unsup)
                _assert_finite("y_full_input",  y_full_input)
                
                # For every (model_hp, ipls_hp) candidate:
                for cv_rmse, cv_rmse_se, cv_r2, model_hp, ipls_hp in hp_results:

                    # get the OUTER selection for this ipls setting (cached)
                    sel, n_intervals_selected, n_features_kept = _get_outer_sel(ipls_hp)

                    # now do the slicing per-combo
                    Xp_full = Xp_full_unsup[sel, :].T.astype(np.float32)  # (n_spectra_total, n_selected_features)
                    X_loo32 = Xp_loo_unsup[sel, :].T.astype(np.float32)   # (9, n_selected_features)

                    _assert_finite("Xp_full", Xp_full)
                    _assert_finite("X_loo32", X_loo32)

                    # ----- CACHE LOOKUP -----
                    sel_sig = _sel_signature(sel.astype(np.int64))
                    cache_key = _cache_key(
                        model_name=model_name,
                        params=model_hp,
                        preprocess=prep_chain,
                        target_col=target_column,
                        train_fp=train_fp,
                        sel_sig=sel_sig,
                    )

                    bundle = load_model_bundle_if_exists(CACHE_DIR, cache_key)
                    if bundle is None:
                        # fit fresh
                        mdl = get_model_by_name(model_name, **model_hp)
                        mdl.fit(Xp_full, y_full_input)

                        # Save fitted preprocessor & model (preprocessor here is the OUTER-FULL one)
                        save_model_bundle(
                            CACHE_DIR, cache_key,
                            preprocessor=prep_full,
                            model=mdl,
                            meta={
                                "model": model_name,
                                "params": _canonicalize_params(model_hp),
                                "preprocess": list(prep_chain),
                                "target": target_column,
                                "train_fp": train_fp,
                                "sel_sig": sel_sig,
                                "ipls_params": (json.dumps(ipls_hp, sort_keys=True) if ipls_hp is not None else "none"),
                                "n_features_kept": int(n_features_kept),
                                "n_intervals_kept": (int(n_intervals_selected) if ipls_hp is not None else None),
                            }
                        )
                    else:
                        mdl = bundle["model"]
                        # Note: you *could* also reuse bundle["preprocessor"], but we already applied `prep_full` above.

                    # re-train score on sample means (if bundle existed, this uses cached mdl)
                    preds_full = mdl.predict(Xp_full)
                    preds_full = preds_full.ravel() if getattr(preds_full, "ndim", 1) > 1 else np.ravel(preds_full)
                    retr_rmse, retr_r2 = _rmse_r2_on_sample_means(y_vals, preds_full, reps=9)

                    # held-out prediction
                    raw_preds = mdl.predict(X_loo32)
                    raw_preds = raw_preds.ravel() if getattr(raw_preds, "ndim", 1) > 1 else np.ravel(raw_preds)
                    test_mean = float(raw_preds.mean())
                    test_std  = float(raw_preds.std(ddof=1))
                    test_true = float(meta_test[target_column].values[held_sample_idx])

                    is_best = (model_hp == best_model_hp) and (ipls_hp == best_ipls_hp)
                    ipls_cols = {
                        'ipls_intervals_kept': n_intervals_selected if ipls_hp is not None else np.nan,
                        'ipls_features_kept':  n_features_kept     if ipls_hp is not None else np.nan,
                        'ipls_params':         json.dumps(ipls_hp, sort_keys=True) if ipls_hp is not None else "none",
                    }

                    detail_rows.append({
                        'fold': fold,
                        'model': model_name,
                        'preprocess': '+'.join(prep_chain),
                        'hyperparams': json.dumps(model_hp, sort_keys=True),
                        'is_best_by_cv_rmse': bool(is_best),
                        'cv_rmse': cv_rmse,
                        'cv_rmse_se': cv_rmse_se,
                        'cv_r2':   cv_r2,
                        'retrain_rmse': retr_rmse,
                        'retrain_r2':   retr_r2,
                        'test_pred_mean': test_mean,
                        'test_true':      test_true,
                        'test_pred_std':  test_std,
                        'pure_nesting': bool(pure_nesting),
                        **ipls_cols
                    })




    # ---- 7) Save detailed per-fold table ----
    df = pd.DataFrame(detail_rows)
    df.to_csv(os.path.join(output_dir, 'loo_per_fold_all_hyperparams.csv'), index=False)
    # ---- 7a) Nested-CV (per-fold best hyperparams) ----
    # One row per (fold, model, preprocess): the combo each outer fold actually chose
    nested_per_fold = (
        df[df['is_best_by_cv_rmse']]
          .copy()
          .sort_values(['model', 'preprocess', 'fold'])
    )
    # ADD: sanity check (each (model, preprocess) should have exactly n_test outer folds)
    assert not (
        nested_per_fold.groupby(['model','preprocess'])['fold'].nunique() != n_test
    ).any(), "Missing nested rows for some (model, preprocess)."
    # (optional) keep a CSV for quick inspection
    nested_per_fold.to_csv(os.path.join(output_dir, 'nested_per_fold_best_hp.csv'), index=False)
    
    # ---- 7b) Aggregate nested-CV across folds (one row per model+preprocess) ----
    def _final_metrics_nested(group):
        cv_rmse_mean      = group['cv_rmse'].mean()
        cv_rmse_se_mean   = group['cv_rmse_se'].mean()
        cv_r2_mean        = group['cv_r2'].mean()
        retrain_rmse_mean = group['retrain_rmse'].mean()
        retrain_r2_mean   = group['retrain_r2'].mean()
        # Held-out performance: compute from per-fold held-out predictions
        final_rmse = float(np.sqrt(mean_squared_error(group['test_true'].values,
                                                      group['test_pred_mean'].values)))
        final_r2   = float(r2_score(group['test_true'].values,
                                    group['test_pred_mean'].values))
        return pd.Series({
            'cv_rmse_mean':        cv_rmse_mean,
            'cv_rmse_se_mean':     cv_rmse_se_mean,
            'cv_r2_mean':          cv_r2_mean,
            'retrain_rmse_mean':   retrain_rmse_mean,
            'retrain_r2_mean':     retrain_r2_mean,
            'final_rmse':          final_rmse,
            'final_r2':            final_r2,
        })
    
    nested_by_combo = (
        nested_per_fold
        .groupby(['model', 'preprocess'], as_index=False)
        .apply(_final_metrics_nested)
        .reset_index(drop=True)
    )
    
    # ---- 7c) (Nice-to-have) Which hyperparams were chosen how often? ----
    # This helps you see stability/consistency of the fold-wise picks.
    nested_hp_counts = (
        nested_per_fold
          .groupby(['model', 'preprocess', 'hyperparams', 'ipls_params'], as_index=False)
          .size()
          .rename(columns={'size': 'chosen_count'})
          .sort_values(['model', 'preprocess', 'chosen_count'], ascending=[True, True, False])
    )
    # ADD: quick CSVs
    nested_by_combo.to_csv(os.path.join(output_dir, 'nested_by_combo.csv'), index=False)
    nested_hp_counts.to_csv(os.path.join(output_dir, 'nested_hp_counts.csv'), index=False)
    
    # ---- 8) Aggregate by (model, preprocess, hyperparams) across all outer folds ----
    def _final_metrics(group):
        # inner-CV and retrain: mean over folds
        cv_rmse_mean       = group['cv_rmse'].mean()
        cv_rmse_se_mean    = group['cv_rmse_se'].mean()   # <-- NEW
        cv_r2_mean         = group['cv_r2'].mean()
        retrain_rmse_mean  = group['retrain_rmse'].mean()
        retrain_r2_mean    = group['retrain_r2'].mean()
    
        # final test performance: computed from held-out preds across folds
        final_rmse = float(np.sqrt(mean_squared_error(group['test_true'].values,
                                                      group['test_pred_mean'].values)))
        final_r2   = float(r2_score(group['test_true'].values,
                                    group['test_pred_mean'].values))
        return pd.Series({
            'cv_rmse_mean':        cv_rmse_mean,
            'cv_rmse_se_mean':     cv_rmse_se_mean,   # <-- NEW
            'cv_r2_mean':          cv_r2_mean,
            'retrain_rmse_mean':   retrain_rmse_mean,
            'retrain_r2_mean':     retrain_r2_mean,
            'final_rmse':          final_rmse,
            'final_r2':            final_r2,
        })

    by_combo = (
        df.groupby(['model','preprocess','hyperparams','ipls_params'], as_index=False)
          .apply(_final_metrics)
          .reset_index(drop=True)
    )

    
    # also keep a compact “best-by-CV” view (optional)
    best_by_cv = (
        by_combo.sort_values(['model','preprocess','cv_rmse_mean'])
                .groupby(['model','preprocess'], as_index=False)
                .first()
    )
        # ===== Parity plots: best GLOBAL vs best ENSEMBLE per model =====
    
    def _parity_plot(df_rows, title, outpath):
        x = df_rows['test_true'].values
        y = df_rows['test_pred_mean'].values
        yerr = df_rows['test_pred_std'].values if 'test_pred_std' in df_rows.columns else None
    
        plt.figure(figsize=(6.2, 6.2))
        plt.errorbar(x, y, yerr=yerr, fmt='o', alpha=0.85, capsize=3)
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', linewidth=1)
    
        rmse = float(np.sqrt(mean_squared_error(x, y)))
        r2   = float(r2_score(x, y))
    
        plt.xlabel('True (sample mean)')
        plt.ylabel('Predicted (sample mean ± 1 SD)')
        plt.title(f"{title}\nRMSE={rmse:.3f}, R²={r2:.3f}")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
    
    def _safe_name(s: str) -> str:
        return ''.join(ch if ch.isalnum() or ch in ('-','_') else '_' for ch in s)
    
    # --- Best GLOBAL per model (single hyperparam set across all folds) ---
    best_global = (
        by_combo.sort_values(['model', 'final_rmse'])
                .groupby('model', as_index=False)
                .first()
    )
    
    for _, row in best_global.iterrows():
        model = row['model']
        prep  = row['preprocess']
        hp    = row['hyperparams']   # JSON string
        ipls  = row['ipls_params']   # JSON string or "none"
    
        mask = (
            (df['model'] == model) &
            (df['preprocess'] == prep) &
            (df['hyperparams'] == hp) &
            (df['ipls_params'] == ipls)
        )
        pts = df.loc[mask, ['test_true','test_pred_mean','test_pred_std']].copy()
        if pts.empty:
            continue
    
        title = f"{model} • GLOBAL\nprep={prep}\nHP={hp}\niPLS={ipls}"
        fname = f"parity_global_{_safe_name(target_column)}_{_safe_name(model)}_{_safe_name(prep)}.png"
        outpath = os.path.join(output_dir, fname)
        _parity_plot(pts, title, outpath)
        print(f"[LOO] Saved parity: {outpath}")

    
    # --- Best ENSEMBLE per model (nested CV: fold-specific hyperparams) ---
    best_ens = (
        nested_by_combo.sort_values(['model', 'final_rmse'])
                       .groupby('model', as_index=False)
                       .first()
    )
    
    for _, row in best_ens.iterrows():
        model = row['model']
        prep  = row['preprocess']
    
        mask = (
            (nested_per_fold['model'] == model) &
            (nested_per_fold['preprocess'] == prep)
        )
        pts = nested_per_fold.loc[mask, ['test_true','test_pred_mean','test_pred_std']].copy()
        if pts.empty:
            continue
    
        title = f"{model} • ENSEMBLE (nested)\nprep={prep}\nHP=ensemble"
        fname = f"parity_ensemble_{_safe_name(target_column)}_{_safe_name(model)}_{_safe_name(prep)}.png"
        outpath = os.path.join(output_dir, fname)
        _parity_plot(pts, title, outpath)
        print(f"[LOO] Saved parity: {outpath}")

    # ---- 9) Save to Excel (multiple sheets) ----
    xlsx_path = os.path.join(output_dir, f"loo_report_{target_column}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        df.to_excel(writer,               sheet_name='per_fold',        index=False)
        by_combo.to_excel(writer,         sheet_name='by_combo',        index=False)
        best_by_cv.to_excel(writer,       sheet_name='best_by_cv',      index=False)
    
        # NEW nested-CV sheets
        nested_per_fold.to_excel(writer,  sheet_name='nested_per_fold', index=False)
        nested_by_combo.to_excel(writer,  sheet_name='nested_by_combo', index=False)
        nested_hp_counts.to_excel(writer, sheet_name='nested_hp_counts', index=False)
    
        
    print(f"[LOO] Wrote detailed Excel report → {xlsx_path}")
    
    # (optional) keep a light CSV with final metrics only
    by_combo.to_csv(os.path.join(output_dir, 'loo_metrics_by_combo.csv'), index=False)
    # --- Write spectral checks log (per model/preprocess, train+test) ---
    prov_path = os.path.join(output_dir, "spectral_checks_log.csv")
    pd.DataFrame(spectral_checks_log).to_csv(prov_path, index=False)
    print(f"[LOO] Wrote spectral checks log → {prov_path}")

    return df, by_combo




