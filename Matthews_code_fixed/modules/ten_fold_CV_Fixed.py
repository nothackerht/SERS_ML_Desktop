# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 15:28:27 2025

@author: spect
"""

# -*- coding: utf-8 -*-
"""
Nested, grouped CV for PLS and Random Forest with *training-only* preprocessing
and sample-level scoring. Compatible with the provided data_loader: it expects
`all_spectra` with shape (n_features=1731, 9*N_samples) and a metadata frame
aligned to samples.

Key properties (per your requirements):
- Hyperparameter SEARCH SPACE is unchanged vs your original code.
- Preprocessing is refit **inside every inner CV fold** (pure nesting).
- Inner CV uses **GroupKFold** with groups = sample IDs (each repeated 9×).
- All model selection and final reporting use **sample-level metrics**
  (i.e., average 9 spectra per sample before scoring).
- Works for PLS (n_components in [2..18]) and RF (your random grid).

Usage sketch (pseudo):

from data_loader import load_data
_, _, all_spectra, meta = load_data(data_dir, metadata_csv, include_types=("DM1","Control"))
X = all_spectra.T           # (9N, 1731) for sklearn
meta = meta.reset_index(drop=True)

rm = NestedGroupedCV(all_spectra=all_spectra, meta=meta)

# PLS
pls_res = rm.run_pls_nested(
    preprocess_methods=["EMSC"],
    n_components_list=list(range(2,19)),
    n_splits_outer=10,
    n_splits_inner=5,
)
print(pls_res["summary"])  # dict with outer CV sample-level RMSE/R2 etc.

# RF
rf_grid = rm.generate_random_rf_params(num_iterations=100)
rf_res = rm.run_rf_nested(
    preprocess_methods=["SNV"],
    rf_param_list=rf_grid,
    n_splits_outer=10,
    n_splits_inner=5,
)
print(rf_res["summary"])  # dict with outer CV sample-level RMSE/R2 etc.

"""
from __future__ import annotations
import os
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from modules.preprocessing import Preprocessing
import random


# --------------------------- helpers ---------------------------
def _groups_for_spectra(n_samples: int, reps: int = 9) -> np.ndarray:
    """Return group labels length=9*N mapping each spectrum to its sample."""
    return np.repeat(np.arange(n_samples, dtype=int), reps)


def _sample_means(y_like: np.ndarray, reps: int = 9) -> np.ndarray:
    """Collapse spectra-level (len = 9*N) vector or matrix to sample means (N, ...)."""
    y = np.asarray(y_like)
    assert y.shape[0] % reps == 0, "length/dim 0 must be multiple of reps"
    N = y.shape[0] // reps
    if y.ndim == 1:
        return y.reshape(N, reps).mean(axis=1)
    return y.reshape(N, reps, -1).mean(axis=1)


def _rmse_r2_sample_level(y_true_s: np.ndarray, y_pred_s: np.ndarray) -> Tuple[float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true_s, y_pred_s)))
    r2 = float(r2_score(y_true_s, y_pred_s))
    return rmse, r2


def _select_via_mse(mse_list: List[float]) -> int:
    """Index of lowest MSE (ties -> first)."""
    return int(np.argmin(np.asarray(mse_list)))

def _make_intervals(n_features: int, num_intervals: int) -> List[Tuple[int,int]]:
    """Evenly split spectral dimension into `num_intervals` [start, end) slices."""
    step = n_features // num_intervals
    edges = [(i*step, (i+1)*step if i < num_intervals-1 else n_features) for i in range(num_intervals)]
    return edges

def _best_interval_grouped(
    X_tr: np.ndarray, Y_tr: np.ndarray, groups_tr: np.ndarray,
    preprocess_methods: List[str], n_components: int, num_intervals: int,
    reps: int, n_splits: int = 3, random_state: int = 42
) -> Tuple[Tuple[int,int], float]:
    """
    Pick the single best interval on *training data only* via grouped CV.
    Returns (start, end) in feature-space (columns of X_tr) and its mean MSE.
    """
    intervals = _make_intervals(X_tr.shape[1], num_intervals)
    inner = GroupKFold(n_splits=min(n_splits, len(np.unique(groups_tr))))
    rng = np.random.default_rng(random_state)  # (not strictly needed; here for parity)

    interval_scores = []
    for (a, b) in intervals:
        fold_mse = []
        for iti, ivo in inner.split(X_tr, groups=groups_tr):
            X_tr_i, X_vl_i = X_tr[iti][:, a:b], X_tr[ivo][:, a:b]
            Y_tr_i, Y_vl_i = Y_tr[iti],          Y_tr[ivo]

            # fit preprocessing on inner-train only, then apply to both
            Xt_tr_i, Xt_vl_i = NestedGroupedCV._fit_transform(
                NestedGroupedCV, X_tr_i, X_vl_i, preprocess_methods
            )

            pls = PLSRegression(n_components=int(n_components))
            pls.fit(Xt_tr_i, Y_tr_i)
            Y_vl_hat = pls.predict(Xt_vl_i)

            # sample-level mse
            Y_vl_s  = _sample_means(Y_vl_i,  reps=reps)
            Y_hat_s = _sample_means(Y_vl_hat, reps=reps)
            fold_mse.append(mean_squared_error(Y_vl_s, Y_hat_s))
        interval_scores.append(np.mean(fold_mse))

    best_idx = int(np.argmin(interval_scores))
    return intervals[best_idx], float(interval_scores[best_idx])


# --------------------------- main class ---------------------------
@dataclass
class NestedGroupedCV:
    all_spectra: np.ndarray  # (1731, 9N)
    meta: pd.DataFrame       # aligned to N samples (rows), includes targets
    reps: int = 9

    # -------- common data prep --------
    def _build_xy(self, target_cols: List[str], include_types: Tuple[str, ...] = ("DM1", "Control")):
        meta0 = self.meta[self.meta["Type"].isin(include_types)].copy().reset_index(drop=True)
        # Controls: same mapping as your original
        meta0.loc[meta0["Type"] == "Control", "target_SI"] = 0
        meta0.loc[meta0["Type"] == "Control", "HGS_pp_avg"] = 100
        meta0.loc[meta0["Type"] == "Control", "ADF_pp_avg"] = 100

        # Keep rows where ADF exists (exactly as your original filter)
        meta0 = meta0.dropna(subset=["ADF_pp_avg"]).reset_index(drop=True)
        N = len(meta0)

        # Indices of spectra columns to keep (9 per kept sample)
        keep_cols = np.concatenate([np.arange(i*self.reps, (i+1)*self.reps) for i in range(N)], dtype=int)
        X_all = self.all_spectra[:, keep_cols].T  # (9N, 1731) sklearn-style

        # Targets in order; repeat each sample 9× to align to X_all rows
        Y_s = meta0[target_cols].to_numpy(dtype=float)             # (N, K)
        Y = np.repeat(Y_s, self.reps, axis=0)                      # (9N, K)

        groups = _groups_for_spectra(N, reps=self.reps)            # (9N,)
        return X_all, Y, Y_s, groups, meta0

    # -------- preprocessing (fit on training-only) --------
    def _fit_transform(self, X_train: np.ndarray, X_apply: np.ndarray, methods: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the unsupervised preprocessing on X_train^T features, then transform both.
        Preprocessing() expects (n_features, n_spectra)."""
        pre = Preprocessing()
        # Fit on training spectra only
        pre.fit(X_train.T, methods)
        Xt_tr = pre.transform(X_train.T, methods).T
        Xt_ap = pre.transform(X_apply.T, methods).T
        return Xt_tr.astype(np.float32), Xt_ap.astype(np.float32)

    # ----------------------- PLS -----------------------
    def run_pls_nested(
        self,
        *,
        preprocess_methods: List[str],
        n_components_list: List[int],
        n_splits_outer: int = 10,
        n_splits_inner: int = 5,
        target_order: List[str] = ("target_SI", "HGS_pp_avg", "ADF_pp_avg"),
        random_state: int = 42,
    ) -> Dict:
        """Nested GroupKFold for multi-target PLS.
        Model family and hyperparameter grid are unchanged vs your original.
        Selection criterion = lowest sample-level MSE (averaged over targets).
        """
        X, Y, Y_s, groups, meta_kept = self._build_xy(list(target_order))
        N = len(meta_kept)

        # Outer split over *samples*
        outer = KFold(n_splits=min(n_splits_outer, N), shuffle=True, random_state=random_state)
        sample_indices = np.arange(N)

        # Collect per-sample held-out predictions (sample-level)
        held_true = []  # (N, 3)
        held_pred = []  # (N, 3)
        chosen_hps = []

        for outer_fold, (tr_s_idx, te_s_idx) in enumerate(outer.split(sample_indices), start=1):
            # Map sample indices -> spectra row indices
            spectra_tr = np.isin(np.repeat(sample_indices, self.reps), tr_s_idx)
            spectra_te = np.isin(np.repeat(sample_indices, self.reps), te_s_idx)

            X_tr, X_te = X[spectra_tr], X[spectra_te]
            Y_tr, Y_te = Y[spectra_tr], Y[spectra_te]

            # ----- INNER GroupKFold over training samples -----
            inner_groups = _groups_for_spectra(len(tr_s_idx), reps=self.reps)
            # remap training spectra in order of spectra_tr
            # Build mapping from spectra_tr rows (which are grouped) to local group ids
            # spectra_tr is constructed by repeating the boolean mask in sample order, so this holds:
            # groups[spectra_tr] is 0..N-1 filtered; we need to compress to 0..len(tr_s_idx)-1 preserving order
            # create a look-up
            tr_sorted = np.sort(tr_s_idx)
            compress = {g: i for i, g in enumerate(tr_sorted)}
            local_groups = np.array([compress[int(g)] for g in groups[spectra_tr]], dtype=int)

            inner = GroupKFold(n_splits=min(n_splits_inner, len(tr_s_idx)))

            hp_mse: List[float] = []
            for n_comp in n_components_list:
                fold_mse = []
                for iti, ivo in inner.split(X_tr, groups=local_groups):
                    X_tr_i, X_vl_i = X_tr[iti], X_tr[ivo]
                    Y_tr_i, Y_vl_i = Y_tr[iti], Y_tr[ivo]

                    # fit preprocessing on inner-TRAIN only, transform train & val
                    Xt_tr_i, Xt_vl_i = self._fit_transform(X_tr_i, X_vl_i, preprocess_methods)

                    pls = PLSRegression(n_components=int(n_comp))
                    pls.fit(Xt_tr_i, Y_tr_i)
                    Y_vl_hat = pls.predict(Xt_vl_i)

                    # sample-level collapse, then MSE over all targets
                    Y_vl_s = _sample_means(Y_vl_i, reps=self.reps)
                    Y_hat_s = _sample_means(Y_vl_hat, reps=self.reps)
                    mse_val = mean_squared_error(Y_vl_s, Y_hat_s)
                    fold_mse.append(float(mse_val))
                hp_mse.append(float(np.mean(fold_mse)))

            best_idx = _select_via_mse(hp_mse)
            best_ncomp = int(n_components_list[best_idx])
            chosen_hps.append({"n_components": best_ncomp})

            # ----- retrain on full training (outer) with chosen HPs -----
            Xt_tr_full, Xt_te_full = self._fit_transform(X_tr, X_te, preprocess_methods)
            pls_best = PLSRegression(n_components=best_ncomp)
            pls_best.fit(Xt_tr_full, Y_tr)
            Y_te_hat = pls_best.predict(Xt_te_full)

            # collapse to sample-level for the held-out samples
            Y_te_s = _sample_means(Y_te, reps=self.reps)
            Y_hat_s = _sample_means(Y_te_hat, reps=self.reps)

            held_true.append(Y_te_s)
            held_pred.append(Y_hat_s)

        Y_true_all = np.vstack(held_true)
        Y_pred_all = np.vstack(held_pred)

        # sample-level overall metrics
        out = {"per_target": {}, "chosen_hyperparams": chosen_hps}
        targets = list(target_order)
        for j, tname in enumerate(targets):
            rmse_j, r2_j = _rmse_r2_sample_level(Y_true_all[:, j], Y_pred_all[:, j])
            out["per_target"][tname] = {"rmse": rmse_j, "r2": r2_j}
        out["summary"] = {
            "rmse_macro": float(np.sqrt(mean_squared_error(Y_true_all, Y_pred_all))),
            "r2_macro": float(r2_score(Y_true_all, Y_pred_all, multioutput="variance_weighted")),
            "n_outer_folds": len(held_true),
        }
        out["heldout_predictions"] = {
            "y_true_sample": Y_true_all,
            "y_pred_sample": Y_pred_all,
        }
        return out

    # ----------------------- Random Forest -----------------------
    def generate_random_rf_params(self, num_iterations: int) -> List[Dict]:
        rf_params_list = []
        for _ in range(num_iterations):
            params = {
                'n_estimators': random.choice([50, 100, 150, 160, 170, 180, 190]),
                'max_features': 'sqrt',
                'max_depth': random.choice([10, 15, 20, 25]),
                'min_samples_split': random.choice([2, 10, 50, 100, 130, 140, 150, 160, 170, 180]),
                'min_samples_leaf': random.choice([1, 10, 30, 40, 47, 48, 50, 55, 60])
            }
            rf_params_list.append(params)
        return rf_params_list

    def run_rf_nested(
        self,
        *,
        preprocess_methods: List[str],
        rf_param_list: List[Dict],
        n_splits_outer: int = 10,
        n_splits_inner: int = 5,
        target_order: List[str] = ("target_SI", "HGS_pp_avg", "ADF_pp_avg"),
        random_state: int = 42,
    ) -> Dict:
        """Nested GroupKFold for Random Forest.
        Keeps your RF hyperparameter space intact and trains separate RF per target.
        Selection criterion = lowest sample-level MSE (macro-averaged over 3 targets).
        """
        X, Y, Y_s, groups, meta_kept = self._build_xy(list(target_order))
        N = len(meta_kept)

        outer = KFold(n_splits=min(n_splits_outer, N), shuffle=True, random_state=random_state)
        sample_indices = np.arange(N)

        held_true, held_pred, chosen_hps = [], [], []

        for outer_fold, (tr_s_idx, te_s_idx) in enumerate(outer.split(sample_indices), start=1):
            spectra_tr = np.isin(np.repeat(sample_indices, self.reps), tr_s_idx)
            spectra_te = np.isin(np.repeat(sample_indices, self.reps), te_s_idx)

            X_tr, X_te = X[spectra_tr], X[spectra_te]
            Y_tr, Y_te = Y[spectra_tr], Y[spectra_te]

            # Local group ids for training spectra
            tr_sorted = np.sort(tr_s_idx)
            compress = {g: i for i, g in enumerate(tr_sorted)}
            local_groups = np.array([compress[int(g)] for g in groups[spectra_tr]], dtype=int)
            inner = GroupKFold(n_splits=min(n_splits_inner, len(tr_s_idx)))

            hp_mse: List[float] = []
            for hp in rf_param_list:
                fold_mse = []
                for iti, ivo in inner.split(X_tr, groups=local_groups):
                    X_tr_i, X_vl_i = X_tr[iti], X_tr[ivo]
                    Y_tr_i, Y_vl_i = Y_tr[iti], Y_tr[ivo]

                    Xt_tr_i, Xt_vl_i = self._fit_transform(X_tr_i, X_vl_i, preprocess_methods)

                    # train 3 independent RFs (one per target)
                    preds_vl = []
                    for j in range(Y_tr_i.shape[1]):
                        rf = RandomForestRegressor(**hp, random_state=random_state)
                        rf.fit(Xt_tr_i, Y_tr_i[:, j])
                        preds_vl.append(rf.predict(Xt_vl_i))
                    Y_vl_hat = np.column_stack(preds_vl)

                    # sample-level mse (macro over targets)
                    Y_vl_s = _sample_means(Y_vl_i, reps=self.reps)
                    Y_hat_s = _sample_means(Y_vl_hat, reps=self.reps)
                    mse_val = mean_squared_error(Y_vl_s, Y_hat_s)
                    fold_mse.append(float(mse_val))
                hp_mse.append(float(np.mean(fold_mse)))

            best_idx = _select_via_mse(hp_mse)
            best_hp = rf_param_list[best_idx]
            chosen_hps.append(best_hp)

            # Refit on full outer train with chosen HP; preprocess fit on all outer train
            Xt_tr_full, Xt_te_full = self._fit_transform(X_tr, X_te, preprocess_methods)

            preds_te = []
            for j in range(Y_tr.shape[1]):
                rf = RandomForestRegressor(**best_hp, random_state=random_state)
                rf.fit(Xt_tr_full, Y_tr[:, j])
                preds_te.append(rf.predict(Xt_te_full))
            Y_te_hat = np.column_stack(preds_te)

            Y_te_s = _sample_means(Y_te, reps=self.reps)
            Y_hat_s = _sample_means(Y_te_hat, reps=self.reps)

            held_true.append(Y_te_s)
            held_pred.append(Y_hat_s)

        Y_true_all = np.vstack(held_true)
        Y_pred_all = np.vstack(held_pred)

        out = {"per_target": {}, "chosen_hyperparams": chosen_hps}
        targets = list(target_order)
        for j, tname in enumerate(targets):
            rmse_j, r2_j = _rmse_r2_sample_level(Y_true_all[:, j], Y_pred_all[:, j])
            out["per_target"][tname] = {"rmse": rmse_j, "r2": r2_j}
        out["summary"] = {
            "rmse_macro": float(np.sqrt(mean_squared_error(Y_true_all, Y_pred_all))),
            "r2_macro": float(r2_score(Y_true_all, Y_pred_all, multioutput="variance_weighted")),
            "n_outer_folds": len(held_true),
        }
        out["heldout_predictions"] = {
            "y_true_sample": Y_true_all,
            "y_pred_sample": Y_pred_all,
        }
        return out
#-----------------------iPLS--------------------------------
    def run_ipls_nested(
        self,
        *,
        preprocess_methods: List[str],
        n_components_list: List[int],
        num_intervals_list: List[int],
        n_splits_outer: int = 10,
        n_splits_inner: int = 5,
        target_order: List[str] = ("target_SI", "HGS_pp_avg", "ADF_pp_avg"),
        random_state: int = 42,
    ) -> Dict:
        """
        Nested GroupKFold for iPLS (single-interval selection).
        - For each candidate (n_components, num_intervals), inner CV selects
          the best single interval on inner-train via grouped CV, then validates
          on inner-val. We average inner MSE across folds to pick the candidate.
        - On the outer held-out set, we re-select the best interval *using only
          the outer-train* (grouped CV), retrain on the whole outer-train within
          that interval, and predict the outer-test.
        - Preprocessing is always fit on the current training spectra only.
        - Metrics are computed at *sample level*.
        """
        X, Y, Y_s, groups, meta_kept = self._build_xy(list(target_order))
        N = len(meta_kept)

        outer = KFold(n_splits=min(n_splits_outer, N), shuffle=True, random_state=random_state)
        sample_indices = np.arange(N)

        held_true, held_pred, chosen_hps = [], [], []

        for outer_fold, (tr_s_idx, te_s_idx) in enumerate(outer.split(sample_indices), start=1):
            # Map sample->spectra rows
            spectra_tr = np.isin(np.repeat(sample_indices, self.reps), tr_s_idx)
            spectra_te = np.isin(np.repeat(sample_indices, self.reps), te_s_idx)

            X_tr_full, X_te_full = X[spectra_tr], X[spectra_te]
            Y_tr_full, Y_te_full = Y[spectra_tr], Y[spectra_te]

            # Local grouped ids for spectra in outer-train
            tr_sorted = np.sort(tr_s_idx)
            compress = {g: i for i, g in enumerate(tr_sorted)}
            groups_tr_full = np.array([compress[int(g)] for g in groups[spectra_tr]], dtype=int)

            # ---------- INNER model/interval selection ----------
            inner = GroupKFold(n_splits=min(n_splits_inner, len(tr_s_idx)))
            candidate_mse = []
            candidate_defs = []

            for n_comp in n_components_list:
                for n_int in num_intervals_list:
                    fold_mse = []
                    for iti, ivo in inner.split(X_tr_full, groups=groups_tr_full):
                        X_tr_i, X_vl_i = X_tr_full[iti], X_tr_full[ivo]
                        Y_tr_i, Y_vl_i = Y_tr_full[iti], Y_tr_full[ivo]
                        groups_tr_i    = groups_tr_full[iti]

                        # pick best interval on inner-train only (grouped)
                        (a, b), _ = _best_interval_grouped(
                            X_tr=X_tr_i, Y_tr=Y_tr_i, groups_tr=groups_tr_i,
                            preprocess_methods=preprocess_methods,
                            n_components=int(n_comp), num_intervals=int(n_int),
                            reps=self.reps, n_splits=max(3, min(5, len(np.unique(groups_tr_i)))),  # small inner-inner grouping
                            random_state=random_state
                        )

                        # fit preprocessing on inner-train interval, validate on inner-val interval
                        Xt_tr_i, Xt_vl_i = self._fit_transform(X_tr_i[:, a:b], X_vl_i[:, a:b], preprocess_methods)

                        pls = PLSRegression(n_components=int(n_comp))
                        pls.fit(Xt_tr_i, Y_tr_i)
                        Y_vl_hat = pls.predict(Xt_vl_i)

                        # sample-level mse
                        Y_vl_s  = _sample_means(Y_vl_i,  reps=self.reps)
                        Y_hat_s = _sample_means(Y_vl_hat, reps=self.reps)
                        fold_mse.append(mean_squared_error(Y_vl_s, Y_hat_s))

                    candidate_mse.append(float(np.mean(fold_mse)))
                    candidate_defs.append({"n_components": int(n_comp), "num_intervals": int(n_int)})

            best_idx = _select_via_mse(candidate_mse)
            best_hp  = candidate_defs[best_idx]
            chosen_hps.append(best_hp)

            # ---------- Outer retrain: re-select interval on full outer-train ----------
            (best_a, best_b), _ = _best_interval_grouped(
                X_tr=X_tr_full, Y_tr=Y_tr_full, groups_tr=groups_tr_full,
                preprocess_methods=preprocess_methods,
                n_components=int(best_hp["n_components"]),
                num_intervals=int(best_hp["num_intervals"]),
                reps=self.reps, n_splits=max(3, min(5, len(np.unique(groups_tr_full)))),
                random_state=random_state
            )

            # Fit preprocessing on full outer-train interval, apply to outer-test interval
            Xt_tr, Xt_te = self._fit_transform(
                X_tr_full[:, best_a:best_b], X_te_full[:, best_a:best_b], preprocess_methods
            )

            pls = PLSRegression(n_components=int(best_hp["n_components"]))
            pls.fit(Xt_tr, Y_tr_full)
            Y_te_hat = pls.predict(Xt_te)

            # Collapse to sample-level
            Y_te_s  = _sample_means(Y_te_full,  reps=self.reps)
            Y_hat_s = _sample_means(Y_te_hat,   reps=self.reps)

            held_true.append(Y_te_s)
            held_pred.append(Y_hat_s)

        # ---------- aggregate ----------
        Y_true_all = np.vstack(held_true)
        Y_pred_all = np.vstack(held_pred)

        out = {"per_target": {}, "chosen_hyperparams": chosen_hps}
        targets = list(target_order)
        for j, tname in enumerate(targets):
            rmse_j, r2_j = _rmse_r2_sample_level(Y_true_all[:, j], Y_pred_all[:, j])
            out["per_target"][tname] = {"rmse": rmse_j, "r2": r2_j}
        out["summary"] = {
            "rmse_macro": float(np.sqrt(mean_squared_error(Y_true_all, Y_pred_all))),
            "r2_macro":   float(r2_score(Y_true_all, Y_pred_all, multioutput="variance_weighted")),
            "n_outer_folds": len(held_true),
        }
        out["heldout_predictions"] = {
            "y_true_sample": Y_true_all,
            "y_pred_sample": Y_pred_all,
        }
        return out
