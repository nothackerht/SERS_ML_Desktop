# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 11:35:30 2025

@author: spect
"""

# --- imports ---
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import KFold, GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed
import random
# TEMP fallback (works today without extra deps): use sklearn MLP as a drop-in
from sklearn.neural_network import MLPRegressor as ACFNNRegressor
# import your preprocessing class
from modules.preprocessing import Preprocessing

# --- helper functions (place them here) ---
from sklearn.metrics import mean_absolute_error, median_absolute_error, explained_variance_score

def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    medae= float(median_absolute_error(y_true, y_pred))
    evs  = float(explained_variance_score(y_true, y_pred))
    return {"rmse": rmse, "r2": r2, "mae": mae, "medae": medae, "evs": evs}

def _groups_for_spectra(n_samples: int, reps: int = 9) -> np.ndarray:
    return np.repeat(np.arange(n_samples, dtype=int), reps)
def _sample_means_stds(y_like: np.ndarray, reps: int = 9):
    """Return (means, stds) collapsing 9× spectra per sample."""
    y = np.asarray(y_like)
    assert y.shape[0] % reps == 0, "length must be multiple of reps"
    N = y.shape[0] // reps
    if y.ndim == 1:
        y3 = y.reshape(N, reps, 1)
        m  = y3.mean(axis=1).ravel()
        s  = y3.std(axis=1)       # ddof=0
        return m, s
    y3 = y.reshape(N, reps, -1)
    return y3.mean(axis=1), y3.std(axis=1)  # (N, K), (N, K)

def _sample_means(y_like: np.ndarray, reps: int = 9) -> np.ndarray:
    y = np.asarray(y_like)
    assert y.shape[0] % reps == 0, "length must be multiple of reps"
    N = y.shape[0] // reps
    if y.ndim == 1:
        return y.reshape(N, reps).mean(axis=1)
    return y.reshape(N, reps, -1).mean(axis=1)

def _rmse_r2_sample_level(y_true_s: np.ndarray, y_pred_s: np.ndarray):
    rmse = float(np.sqrt(mean_squared_error(y_true_s, y_pred_s)))
    r2   = float(r2_score(y_true_s, y_pred_s))
    return rmse, r2

def _make_intervals(n_features: int, num_intervals: int):
    step = n_features // num_intervals
    return [(i*step, (i+1)*step if i < num_intervals-1 else n_features) for i in range(num_intervals)]

def _best_interval_grouped(
    X_tr: np.ndarray, Y_tr: np.ndarray, groups_tr: np.ndarray,
    preprocess_methods, n_components: int, num_intervals: int,
    reps: int, n_splits: int = 3, random_state: int = 42
):
    intervals = _make_intervals(X_tr.shape[1], num_intervals)
    inner = GroupKFold(n_splits=min(n_splits, len(np.unique(groups_tr))))
    interval_mse = []

    def _fit_transform_local(X_train, X_apply, methods):
        Xtr = X_train.T.copy()
        Xap = X_apply.T.copy()
        for m in (methods or []):
            if m == "EMSC":
                ref = np.mean(Xtr, axis=1)
                Xtr = Preprocessing(Xtr).emsc(Xtr, reference=ref)
                Xap = Preprocessing(Xap).emsc(Xap, reference=ref)
            elif m == "Normalization":
                Xtr = Preprocessing(Xtr).normalize_spectrum(Xtr)
                Xap = Preprocessing(Xap).normalize_spectrum(Xap)
            elif m == "SNV":
                Xtr = Preprocessing(Xtr).snv(Xtr)
                Xap = Preprocessing(Xap).snv(Xap)
            elif m == "Second Derivative":
                Xtr = Preprocessing(Xtr).second_derivative(Xtr)
                Xap = Preprocessing(Xap).second_derivative(Xap)
            else:
                raise ValueError(f"Unknown preprocessing method: {m}")
        return Xtr.T.astype(np.float32), Xap.T.astype(np.float32)

    for (a, b) in intervals:
        fold_mse = []
        for iti, ivo in inner.split(X_tr, groups=groups_tr):
            Xt_i_tr, Xt_i_vl = X_tr[iti][:, a:b], X_tr[ivo][:, a:b]
            Yt_i_tr, Yt_i_vl = Y_tr[iti],          Y_tr[ivo]
            Xt_tr_p, Xt_vl_p = _fit_transform_local(Xt_i_tr, Xt_i_vl, preprocess_methods)

            pls = PLSRegression(n_components=int(n_components))
            pls.fit(Xt_tr_p, Yt_i_tr)
            Y_vl_hat = pls.predict(Xt_vl_p)

            Y_vl_s  = _sample_means(Yt_i_vl,  reps=reps)
            Y_hat_s = _sample_means(Y_vl_hat, reps=reps)
            fold_mse.append(mean_squared_error(Y_vl_s, Y_hat_s))
        interval_mse.append(float(np.mean(fold_mse)))

    best_idx = int(np.argmin(interval_mse))
    return (intervals[best_idx], float(interval_mse[best_idx]))

@dataclass
class GlobalGroupedCV:  # new class to keep your nested one intact; or merge if you prefer
    all_spectra: np.ndarray  # (1731, 9N)
    meta: pd.DataFrame       # N rows (samples)
    reps: int = 9

    # ---- same _build_xy and _fit_transform as in your code ----
    def _build_xy(self, target_cols: List[str], include_types=("DM1","Control")):
        meta0 = self.meta[self.meta["Type"].isin(include_types)].copy().reset_index(drop=True)
        meta0.loc[meta0["Type"] == "Control", "target_SI"]  = 0
        meta0.loc[meta0["Type"] == "Control", "HGS_pp_avg"] = 100
        meta0.loc[meta0["Type"] == "Control", "ADF_pp_avg"] = 100
        meta0 = meta0.dropna(subset=["ADF_pp_avg"]).reset_index(drop=True)
        N = len(meta0)

        keep_cols = np.concatenate([np.arange(i*self.reps, (i+1)*self.reps) for i in range(N)], dtype=int)
        X_all = self.all_spectra[:, keep_cols].T  # (9N, 1731)

        Y_s = meta0[ list(target_cols) ].to_numpy(dtype=float)   # (N, K)
        Y   = np.repeat(Y_s, self.reps, axis=0)                  # (9N, K)
        groups = _groups_for_spectra(N, reps=self.reps)          # (9N,)
        return X_all, Y, Y_s, groups, meta0

    def _fit_transform(self, X_train: np.ndarray, X_apply: np.ndarray, methods: list):
        Xtr = X_train.T.copy()
        Xap = X_apply.T.copy()
        for m in (methods or []):
            if m == "EMSC":
                ref = np.mean(Xtr, axis=1)
                Xtr = Preprocessing(Xtr).emsc(Xtr, reference=ref)
                Xap = Preprocessing(Xap).emsc(Xap, reference=ref)
            elif m == "Normalization":
                Xtr = Preprocessing(Xtr).normalize_spectrum(Xtr)
                Xap = Preprocessing(Xap).normalize_spectrum(Xap)
            elif m == "SNV":
                Xtr = Preprocessing(Xtr).snv(Xtr)
                Xap = Preprocessing(Xap).snv(Xap)
            elif m == "Second Derivative":
                Xtr = Preprocessing(Xtr).second_derivative(Xtr)
                Xap = Preprocessing(Xap).second_derivative(Xap)
            else:
                raise ValueError(f"Unknown preprocessing method: {m}")
        return Xtr.T.astype(np.float32), Xap.T.astype(np.float32)

    # ---------------- PLS (GLOBAL HPs) ----------------
    def run_pls_global(
        self,
        *,
        preprocess_methods: List[str],
        n_components_list: List[int],
        n_splits: int = 10,
        target_order: List[str] = ("target_SI","HGS_pp_avg","ADF_pp_avg"),
        random_state: int = 42,
    ) -> Dict:
        X, Y, Y_s, groups, meta_kept = self._build_xy(list(target_order))
        N = len(meta_kept)
        outer = KFold(n_splits=min(n_splits, N), shuffle=True, random_state=random_state)
        sample_indices = np.arange(N)
    
        def score_pls(nc: int) -> float:
            fold_mses = []
            for tr_s_idx, te_s_idx in outer.split(sample_indices):
                spectra_tr = np.isin(np.repeat(sample_indices, self.reps), tr_s_idx)
                spectra_te = np.isin(np.repeat(sample_indices, self.reps), te_s_idx)
                X_tr, X_te = X[spectra_tr], X[spectra_te]
                Y_tr, Y_te = Y[spectra_tr], Y[spectra_te]
                Xt_tr, Xt_te = self._fit_transform(X_tr, X_te, preprocess_methods)
                pls = PLSRegression(n_components=int(nc))
                pls.fit(Xt_tr, Y_tr)
                Y_hat = pls.predict(Xt_te)
                Y_te_s,  _ = _sample_means_stds(Y_te,  reps=self.reps)
                Y_hat_s, _ = _sample_means_stds(Y_hat, reps=self.reps)
                fold_mses.append(mean_squared_error(Y_te_s, Y_hat_s))
            return float(np.mean(fold_mses))
    
        hp_mse = [score_pls(nc) for nc in n_components_list]
        best_idx = int(np.argmin(hp_mse))
        best_nc  = int(n_components_list[best_idx])
    
        held_true, held_pred = [], []
        held_true_std, held_pred_std = [], []
        for tr_s_idx, te_s_idx in KFold(n_splits=min(n_splits, N), shuffle=True, random_state=random_state).split(sample_indices):
            spectra_tr = np.isin(np.repeat(sample_indices, self.reps), tr_s_idx)
            spectra_te = np.isin(np.repeat(sample_indices, self.reps), te_s_idx)
            X_tr, X_te = X[spectra_tr], X[spectra_te]
            Y_tr, Y_te = Y[spectra_tr], Y[spectra_te]
            Xt_tr, Xt_te = self._fit_transform(X_tr, X_te, preprocess_methods)
            pls = PLSRegression(n_components=best_nc)
            pls.fit(Xt_tr, Y_tr)
            Y_hat = pls.predict(Xt_te)
    
            m_true, s_true = _sample_means_stds(Y_te,  reps=self.reps)
            m_pred, s_pred = _sample_means_stds(Y_hat, reps=self.reps)
            held_true.append(m_true);     held_pred.append(m_pred)
            held_true_std.append(s_true); held_pred_std.append(s_pred)
    
        Y_true_all     = np.vstack(held_true)
        Y_pred_all     = np.vstack(held_pred)
        Y_true_std_all = np.vstack(held_true_std)
        Y_pred_std_all = np.vstack(held_pred_std)
    
        out = {"per_target": {}, "chosen_hyperparams": {"n_components": best_nc, "preprocess": preprocess_methods}}
        for j, t in enumerate(target_order):
            rmse_j, r2_j = _rmse_r2_sample_level(Y_true_all[:, j], Y_pred_all[:, j])
            out["per_target"][t] = {"rmse": rmse_j, "r2": r2_j}
        out["summary"] = {
            "rmse_macro": float(np.sqrt(mean_squared_error(Y_true_all, Y_pred_all))),
            "r2_macro":   float(r2_score(Y_true_all, Y_pred_all, multioutput="variance_weighted")),
            "hp_mse_curve": dict(zip(map(int, n_components_list), map(float, hp_mse))),
            "n_folds": int(min(n_splits, N)),
        }
        out["heldout_predictions"] = {
            "y_true_sample": Y_true_all,
            "y_pred_sample": Y_pred_all,
            "y_true_sample_std": Y_true_std_all,   # <-- x-error bars
            "y_pred_sample_std": Y_pred_std_all,   # <-- y-error bars
        }
        return out

    # ------------- Random Forest (GLOBAL HPs) -------------
    def run_rf_global(
        self,
        *,
        preprocess_methods: List[str],
        rf_param_list: List[Dict],
        n_splits: int = 10,
        target_order: List[str] = ("target_SI","HGS_pp_avg","ADF_pp_avg"),
        random_state: int = 42,
    ) -> Dict:
        X, Y, Y_s, groups, meta_kept = self._build_xy(list(target_order))
        N = len(meta_kept)
        outer = KFold(n_splits=min(n_splits, N), shuffle=True, random_state=random_state)
        sample_indices = np.arange(N)
    
        def score_rf(hp: Dict) -> float:
            fold_mses = []
            for tr_s_idx, te_s_idx in outer.split(sample_indices):
                spectra_tr = np.isin(np.repeat(sample_indices, self.reps), tr_s_idx)
                spectra_te = np.isin(np.repeat(sample_indices, self.reps), te_s_idx)
                X_tr, X_te = X[spectra_tr], X[spectra_te]
                Y_tr, Y_te = Y[spectra_tr], Y[spectra_te]
                Xt_tr, Xt_te = self._fit_transform(X_tr, X_te, preprocess_methods)
    
                preds = []
                for j in range(Y_tr.shape[1]):
                    rf = RandomForestRegressor(**hp, random_state=random_state, n_jobs=-1)
                    rf.fit(Xt_tr, Y_tr[:, j])
                    preds.append(rf.predict(Xt_te))
                Y_hat = np.column_stack(preds)
    
                Y_te_s,  _ = _sample_means_stds(Y_te,  reps=self.reps)
                Y_hat_s, _ = _sample_means_stds(Y_hat, reps=self.reps)
                fold_mses.append(mean_squared_error(Y_te_s, Y_hat_s))
            return float(np.mean(fold_mses))
    
        hp_mse = [score_rf(hp) for hp in rf_param_list]
        best_idx = int(np.argmin(hp_mse))
        best_hp  = rf_param_list[best_idx]
    
        held_true, held_pred = [], []
        held_true_std, held_pred_std = [], []
        for tr_s_idx, te_s_idx in KFold(n_splits=min(n_splits, N), shuffle=True, random_state=random_state).split(sample_indices):
            spectra_tr = np.isin(np.repeat(sample_indices, self.reps), tr_s_idx)
            spectra_te = np.isin(np.repeat(sample_indices, self.reps), te_s_idx)
            X_tr, X_te = X[spectra_tr], X[spectra_te]
            Y_tr, Y_te = Y[spectra_tr], Y[spectra_te]
            Xt_tr, Xt_te = self._fit_transform(X_tr, X_te, preprocess_methods)
    
            preds = []
            for j in range(Y_tr.shape[1]):
                rf = RandomForestRegressor(**best_hp, random_state=random_state, n_jobs=-1)
                rf.fit(Xt_tr, Y_tr[:, j])
                preds.append(rf.predict(Xt_te))
            Y_hat = np.column_stack(preds)
    
            m_true, s_true = _sample_means_stds(Y_te,  reps=self.reps)
            m_pred, s_pred = _sample_means_stds(Y_hat, reps=self.reps)
            held_true.append(m_true);     held_pred.append(m_pred)
            held_true_std.append(s_true); held_pred_std.append(s_pred)
    
        Y_true_all     = np.vstack(held_true)
        Y_pred_all     = np.vstack(held_pred)
        Y_true_std_all = np.vstack(held_true_std)
        Y_pred_std_all = np.vstack(held_pred_std)
    
        out = {"per_target": {}, "chosen_hyperparams": {"rf_params": best_hp, "preprocess": preprocess_methods}}
        for j, t in enumerate(target_order):
            rmse_j, r2_j = _rmse_r2_sample_level(Y_true_all[:, j], Y_pred_all[:, j])
            out["per_target"][t] = {"rmse": rmse_j, "r2": r2_j}
        out["summary"] = {
            "rmse_macro": float(np.sqrt(mean_squared_error(Y_true_all, Y_pred_all))),
            "r2_macro":   float(r2_score(Y_true_all, Y_pred_all, multioutput="variance_weighted")),
            "hp_mse_list": [float(x) for x in hp_mse],
            "n_folds": int(min(n_splits, N)),
        }
        out["heldout_predictions"] = {
            "y_true_sample": Y_true_all,
            "y_pred_sample": Y_pred_all,
            "y_true_sample_std": Y_true_std_all,
            "y_pred_sample_std": Y_pred_std_all,
        }
        return out

    # ---------------- iPLS (GLOBAL HPs) ----------------
    def run_ipls_global(
        self,
        *,
        preprocess_methods: List[str],
        n_components_list: List[int],
        num_intervals_list: List[int],
        n_splits: int = 10,
        target_order: List[str] = ("target_SI","HGS_pp_avg","ADF_pp_avg"),
        random_state: int = 42,
    ) -> Dict:
        X, Y, Y_s, groups, meta_kept = self._build_xy(list(target_order))
        N = len(meta_kept)
        outer = KFold(n_splits=min(n_splits, N), shuffle=True, random_state=random_state)
        sample_indices = np.arange(N)
    
        def score_ipls(n_comp: int, n_int: int) -> float:
            fold_mses = []
            for tr_s_idx, te_s_idx in outer.split(sample_indices):
                spectra_tr = np.isin(np.repeat(sample_indices, self.reps), tr_s_idx)
                spectra_te = np.isin(np.repeat(sample_indices, self.reps), te_s_idx)
                X_tr, X_te = X[spectra_tr], X[spectra_te]
                Y_tr, Y_te = Y[spectra_tr], Y[spectra_te]
    
                tr_sorted = np.sort(tr_s_idx)
                compress = {g: i for i, g in enumerate(tr_sorted)}
                groups_tr = np.array([compress[int(g)] for g in groups[spectra_tr]], dtype=int)
    
                (a,b), _ = _best_interval_grouped(
                    X_tr=X_tr, Y_tr=Y_tr, groups_tr=groups_tr,
                    preprocess_methods=preprocess_methods,
                    n_components=int(n_comp), num_intervals=int(n_int),
                    reps=self.reps, n_splits=max(3, min(5, len(np.unique(groups_tr)))),
                    random_state=random_state
                )
    
                Xt_tr, Xt_te = self._fit_transform(X_tr[:, a:b], X_te[:, a:b], preprocess_methods)
                pls = PLSRegression(n_components=int(n_comp))
                pls.fit(Xt_tr, Y_tr)
                Y_hat = pls.predict(Xt_te)
    
                Y_te_s,  _ = _sample_means_stds(Y_te,  reps=self.reps)
                Y_hat_s, _ = _sample_means_stds(Y_hat, reps=self.reps)
                fold_mses.append(mean_squared_error(Y_te_s, Y_hat_s))
            return float(np.mean(fold_mses))
    
        candidates = [(int(c), int(I)) for c in n_components_list for I in num_intervals_list]
        cand_mse   = [score_ipls(c, I) for (c, I) in candidates]
        best_idx   = int(np.argmin(cand_mse))
        best_c, best_I = candidates[best_idx]
    
        held_true, held_pred = [], []
        held_true_std, held_pred_std = [], []
        for tr_s_idx, te_s_idx in KFold(n_splits=min(n_splits, N), shuffle=True, random_state=random_state).split(sample_indices):
            spectra_tr = np.isin(np.repeat(sample_indices, self.reps), tr_s_idx)
            spectra_te = np.isin(np.repeat(sample_indices, self.reps), te_s_idx)
            X_tr, X_te = X[spectra_tr], X[spectra_te]
            Y_tr, Y_te = Y[spectra_tr], Y[spectra_te]
    
            tr_sorted = np.sort(tr_s_idx)
            compress = {g: i for i, g in enumerate(tr_sorted)}
            groups_tr = np.array([compress[int(g)] for g in groups[spectra_tr]], dtype=int)
    
            (a,b), _ = _best_interval_grouped(
                X_tr=X_tr, Y_tr=Y_tr, groups_tr=groups_tr,
                preprocess_methods=preprocess_methods,
                n_components=best_c, num_intervals=best_I,
                reps=self.reps, n_splits=max(3, min(5, len(np.unique(groups_tr)))),
                random_state=random_state
            )
    
            Xt_tr, Xt_te = self._fit_transform(X_tr[:, a:b], X_te[:, a:b], preprocess_methods)
            pls = PLSRegression(n_components=best_c)
            pls.fit(Xt_tr, Y_tr)
            Y_hat = pls.predict(Xt_te)
    
            m_true, s_true = _sample_means_stds(Y_te,  reps=self.reps)
            m_pred, s_pred = _sample_means_stds(Y_hat, reps=self.reps)
            held_true.append(m_true);     held_pred.append(m_pred)
            held_true_std.append(s_true); held_pred_std.append(s_pred)
    
        Y_true_all     = np.vstack(held_true)
        Y_pred_all     = np.vstack(held_pred)
        Y_true_std_all = np.vstack(held_true_std)
        Y_pred_std_all = np.vstack(held_pred_std)
    
        out = {"per_target": {}, "chosen_hyperparams": {"n_components": best_c, "num_intervals": best_I, "preprocess": preprocess_methods}}
        for j, t in enumerate(target_order):
            rmse_j, r2_j = _rmse_r2_sample_level(Y_true_all[:, j], Y_pred_all[:, j])
            out["per_target"][t] = {"rmse": rmse_j, "r2": r2_j}
        out["summary"] = {
            "rmse_macro": float(np.sqrt(mean_squared_error(Y_true_all, Y_pred_all))),
            "r2_macro":   float(r2_score(Y_true_all, Y_pred_all, multioutput="variance_weighted")),
            "hp_mse_table": {f"C={c},I={I}": float(m) for (c,I),m in zip(candidates, cand_mse)},
            "n_folds": int(min(n_splits, N)),
        }
        out["heldout_predictions"] = {
            "y_true_sample": Y_true_all,
            "y_pred_sample": Y_pred_all,
            "y_true_sample_std": Y_true_std_all,
            "y_pred_sample_std": Y_pred_std_all,
        }
        return out
    def run_acfnn_global(
        self,
        *,
        preprocess_methods: List[str],
        acfnn_param_list: List[Dict],
        n_splits: int = 10,
        target_order: List[str] = ("target_SI","HGS_pp_avg","ADF_pp_avg"),
        random_state: int = 42,
    ) -> Dict:
        """
        Global HP selection for AC-FNN (multi-target regressor).
        Same contract as PLS/RF: pick one HP set by lowest sample-level MSE,
        return mean/std per-sample so parity plots can draw error bars.
        """
        X, Y, Y_s, groups, meta_kept = self._build_xy(list(target_order))
        N   = len(meta_kept)
        kf  = KFold(n_splits=min(n_splits, N), shuffle=True, random_state=random_state)
        idx = np.arange(N)
    
        def score_hp(hp: Dict) -> float:
            fold_mses = []
            for tr_s_idx, te_s_idx in kf.split(idx):
                spectra_tr = np.isin(np.repeat(idx, self.reps), tr_s_idx)
                spectra_te = np.isin(np.repeat(idx, self.reps), te_s_idx)
                X_tr, X_te = X[spectra_tr], X[spectra_te]
                Y_tr, Y_te = Y[spectra_tr], Y[spectra_te]
    
                Xt_tr, Xt_te = self._fit_transform(X_tr, X_te, preprocess_methods)
    
                # one multi-output regressor (K targets)
                model = ACFNNRegressor(random_state=random_state, **hp)
                model.fit(Xt_tr, Y_tr)
                Y_hat = model.predict(Xt_te)
    
                Y_te_s,  _ = _sample_means_stds(Y_te,  reps=self.reps)
                Y_hat_s, _ = _sample_means_stds(Y_hat, reps=self.reps)
                fold_mses.append(mean_squared_error(Y_te_s, Y_hat_s))
            return float(np.mean(fold_mses))
    
        hp_mse  = [score_hp(hp) for hp in acfnn_param_list]
        best_ix = int(np.argmin(hp_mse))
        best_hp = acfnn_param_list[best_ix]
    
        # Held-out predictions (and stds) with the chosen global HPs
        held_true, held_pred = [], []
        held_true_std, held_pred_std = [], []
        for tr_s_idx, te_s_idx in kf.split(idx):
            spectra_tr = np.isin(np.repeat(idx, self.reps), tr_s_idx)
            spectra_te = np.isin(np.repeat(idx, self.reps), te_s_idx)
            X_tr, X_te = X[spectra_tr], X[spectra_te]
            Y_tr, Y_te = Y[spectra_tr], Y[spectra_te]
    
            Xt_tr, Xt_te = self._fit_transform(X_tr, X_te, preprocess_methods)
            model = ACFNNRegressor(random_state=random_state, **best_hp)
            model.fit(Xt_tr, Y_tr)
            Y_hat = model.predict(Xt_te)
    
            m_true, s_true = _sample_means_stds(Y_te,  reps=self.reps)
            m_pred, s_pred = _sample_means_stds(Y_hat, reps=self.reps)
            held_true.append(m_true);     held_pred.append(m_pred)
            held_true_std.append(s_true); held_pred_std.append(s_pred)
    
        Y_true_all     = np.vstack(held_true)
        Y_pred_all     = np.vstack(held_pred)
        Y_true_std_all = np.vstack(held_true_std)
        Y_pred_std_all = np.vstack(held_pred_std)
    
        out = {"per_target": {}, "chosen_hyperparams": {"ac_fnn_params": best_hp, "preprocess": preprocess_methods}}
        for j, t in enumerate(target_order):
            rmse_j, r2_j = _rmse_r2_sample_level(Y_true_all[:, j], Y_pred_all[:, j])
            out["per_target"][t] = {"rmse": rmse_j, "r2": r2_j}
        out["summary"] = {
            "rmse_macro": float(np.sqrt(mean_squared_error(Y_true_all, Y_pred_all))),
            "r2_macro":   float(r2_score(Y_true_all, Y_pred_all, multioutput="variance_weighted")),
            "hp_mse_list": [float(x) for x in hp_mse],
            "n_folds": int(min(n_splits, N)),
        }
        out["heldout_predictions"] = {
            "y_true_sample": Y_true_all,
            "y_pred_sample": Y_pred_all,
            "y_true_sample_std": Y_true_std_all,
            "y_pred_sample_std": Y_pred_std_all,
        }
        return out
