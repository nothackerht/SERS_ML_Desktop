# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 19:39:16 2025

@author: notha
"""

# -*- coding: utf-8 -*-
"""
External Test (Box 3) evaluator with Group-aware Nested CV model selection on Boxes 1–2.

- Train set: Boxes 1–2 (with/without controls via --include_types)
- Test set : Box 3 (external)
- Selection: GroupKFold nested CV on TRAIN ONLY with 1-SE toward simplicity
- Leakage:   Train-only preprocessing; iPLS intervals chosen on train only
- Models:    PLS / RF / iPLS / AC-FNN(sklearn MLP fallback)
- Outputs:   For each target -> CSV of per-(model,preproc) selected HP + CV & External metrics;
             parity plot + tidy predictions for the winner.

Compatible with main.py:
    python evaluate_external_test_all_combos.py \
        --train_data_dir ... --train_meta_path ... \
        --test_data_dir ...  --test_meta_path  ... \
        --include_types "DM1,Control" \
        --out_dir ...
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import re
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, median_absolute_error, explained_variance_score,
)

# local modules (match your ten_fold module import style)
from data_loader import load_data
from preprocessing import Preprocessing
from plot_parity import parity_plot_sample_level, save_outer_predictions_excel


# ====================== Small utilities ======================

REPS = 9  # spectra per sample

def _metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "rmse":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":    float(r2_score(y_true, y_pred)),
        "mae":   float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "evs":   float(explained_variance_score(y_true, y_pred)),
    }

def _groups_for_spectra(n_samples: int, reps: int = REPS) -> np.ndarray:
    # map spectra rows (reps*N) to their sample id (0..N-1)
    return np.repeat(np.arange(n_samples, dtype=int), reps)

def _sample_means_stds(y_like: np.ndarray, reps: int = REPS):
    """Return (means, stds) collapsing reps spectra per sample. Works with (reps*N,) or (reps*N,K)."""
    y = np.asarray(y_like)
    assert (y.shape[0] % reps) == 0, "length must be multiple of REPS"
    N = y.shape[0] // reps
    if y.ndim == 1:
        y3 = y.reshape(N, reps, 1)
        return y3.mean(axis=1).ravel(), y3.std(axis=1).ravel()
    y3 = y.reshape(N, reps, -1)
    return y3.mean(axis=1), y3.std(axis=1)

def _safe_methods_path(methods):
    return "+".join(m.replace(" ", "_") for m in (methods or [])) if methods else "none"

def _rf_tag(hp: dict) -> str:
    return (f"ne{hp['n_estimators']}_md{hp['max_depth']}_"
            f"mss{hp['min_samples_split']}_msl{hp['min_samples_leaf']}_mf{hp['max_features']}")

def _ac_tag(hp: dict) -> str:
    hls = "-".join(map(str, hp["hidden_layer_sizes"])) if isinstance(hp["hidden_layer_sizes"], (list, tuple)) else str(hp["hidden_layer_sizes"])
    return (f"hls{hls}_act{hp.get('activation','relu')}_bs{hp.get('batch_size','NA')}"
            f"_lr{hp.get('learning_rate_init','NA')}_wd{hp.get('alpha','0')}_mi{hp.get('max_iter','NA')}")

class TorchRegressor:
    """
    Simple feedforward regressor with GPU support.
    Expects a single target column.
    API subset: .fit(X, y), .predict(X) returning shape (n,1)
    """

    def __init__(
        self,
        hidden_layer_sizes=(256, 128),
        activation="relu",
        learning_rate_init=1e-3,
        alpha=1e-4,        # L2 weight decay
        max_iter=200,
        batch_size=64,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None
        self.input_dim_ = None

    def _build_mlp(self, in_dim):
        act_layer = nn.ReLU if self.activation == "relu" else nn.Tanh
        layers = []
        last = in_dim
        for h in self.hidden_layer_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(act_layer())
            last = h
        layers.append(nn.Linear(last, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.input_dim_ = X.shape[1]
        self.model_ = self._build_mlp(self.input_dim_).to(self.device)

        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate_init,
            weight_decay=self.alpha,
        )
        loss_fn = nn.MSELoss()

        n = X.shape[0]
        idx_all = np.arange(n)

        for epoch in range(self.max_iter):
            rng.shuffle(idx_all)
            for start in range(0, n, self.batch_size):
                batch_idx = idx_all[start:start+self.batch_size]
                xb = torch.from_numpy(X[batch_idx]).to(self.device)
                yb = torch.from_numpy(y[batch_idx]).to(self.device)

                optimizer.zero_grad()
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        xb = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            pred = self.model_(xb).cpu().numpy()
        return pred  # shape (n,1)

# ====================== iPLS interval helpers (grouped) ======================

def _make_intervals(n_features: int, num_intervals: int):
    step = max(1, n_features // num_intervals)
    return [(i*step, (i+1)*step if i < num_intervals-1 else n_features) for i in range(num_intervals)]

def _fit_transform_pair(X_train_2D: np.ndarray, X_apply_2D: np.ndarray, methods: list):
    """Fit preprocessing on TRAIN only, apply to TRAIN & APPLY."""
    Xtr = X_train_2D.T.copy()  # (features, n_tr)
    Xap = X_apply_2D.T.copy()  # (features, n_ap)

    for m in (methods or []):
        if m == "EMSC":
            ref = np.mean(Xtr, axis=1)  # train-only ref
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
        elif m in ("", "No Preprocessing", "No preprocessing", "no preprocessing"):
            pass
        else:
            raise ValueError(f"Unknown preprocessing method: {m}")

    return Xtr.T.astype(np.float32), Xap.T.astype(np.float32)

def _best_interval_grouped(
    X_tr: np.ndarray, Y_tr: np.ndarray, groups_tr: np.ndarray,
    preprocess_methods, n_components: int, num_intervals: int,
    reps: int = REPS, n_splits: int = 3
):
    """
    Choose best contiguous interval via GroupKFold on the outer-train ONLY.
    Returns (best_interval (a,b), best_mse).
    """
    intervals = _make_intervals(X_tr.shape[1], num_intervals)
    inner = GroupKFold(n_splits=min(n_splits, len(np.unique(groups_tr))))
    interval_mse = []

    for (a, b) in intervals:
        fold_mse = []
        for iti, ivo in inner.split(X_tr, groups=groups_tr):
            Xt_i_tr, Xt_i_vl = X_tr[iti][:, a:b], X_tr[ivo][:, a:b]
            Yt_i_tr, Yt_i_vl = Y_tr[iti],          Y_tr[ivo]
            Xt_tr_p, Xt_vl_p = _fit_transform_pair(Xt_i_tr, Xt_i_vl, preprocess_methods)

            pls = PLSRegression(n_components=int(n_components))
            pls.fit(Xt_tr_p, Yt_i_tr)
            Y_vl_hat = pls.predict(Xt_vl_p)

            Y_vl_s  = _sample_means_stds(Yt_i_vl,  reps=reps)[0]
            Y_hat_s = _sample_means_stds(Y_vl_hat, reps=reps)[0]
            fold_mse.append(mean_squared_error(Y_vl_s, Y_hat_s))
        interval_mse.append(float(np.mean(fold_mse)))

    best_idx = int(np.argmin(interval_mse))
    return (intervals[best_idx], float(interval_mse[best_idx]))


# ====================== Core class ======================

@dataclass
class GlobalGrouped:
    all_spectra: np.ndarray  # (features, reps*N)
    meta: pd.DataFrame       # N rows (samples)
    reps: int = REPS

    def build_xy(self, target_col: str, include_types=("DM1", "Control")):
        meta0 = self.meta[self.meta["Type"].isin(include_types)].copy().reset_index(drop=True)

        # If your design uses default values for Controls, set here as needed:
        if "Control" in include_types:
            meta0.loc[meta0["Type"] == "Control", "target_SI"]  = 0
            meta0.loc[meta0["Type"] == "Control", "HGS_pp_avg"] = 100
            meta0.loc[meta0["Type"] == "Control", "ADF_pp_avg"] = 100

        meta0 = meta0.dropna(subset=[target_col]).reset_index(drop=True)
        N = len(meta0)

        keep_cols = np.concatenate([np.arange(i*self.reps, (i+1)*self.reps) for i in range(N)], dtype=int)
        X_all = self.all_spectra[:, keep_cols].T  # (reps*N, features)

        y_s = meta0[target_col].to_numpy(dtype=float).reshape(-1, 1)  # (N,1)
        Y   = np.repeat(y_s, self.reps, axis=0)                       # (reps*N,1)
        groups = _groups_for_spectra(N, reps=self.reps)               # (reps*N,)

        return X_all, Y, y_s.ravel(), groups, meta0


# ====================== CLI runner ======================

if __name__ == "__main__":
    import argparse
    import random as _random

    # ---------- parse CLI ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data_dir", required=True)
    ap.add_argument("--train_meta_path", required=True)
    ap.add_argument("--test_data_dir", required=True)
    ap.add_argument("--test_meta_path", required=True)
    ap.add_argument("--include_types", default="DM1,Control")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--outer_folds", type=int, default=10)
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    INCLUDE_TYPES = tuple([s.strip() for s in (args.include_types or "").split(",") if s.strip()])
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- load data ----------
    wn_tr, avg_tr, all_tr, meta_tr = load_data(
        data_dir=args.train_data_dir,
        metadata_path=args.train_meta_path,
        include_types=INCLUDE_TYPES,
        return_filenames=False,
        strict=False,
        report_samples=5,
    )
    wn_te, avg_te, all_te, meta_te = load_data(
        data_dir=args.test_data_dir,
        metadata_path=args.test_meta_path,
        include_types=INCLUDE_TYPES,
        return_filenames=False,
        strict=False,
        report_samples=5,
    )
    # ---------- ensure numeric targets in TEST metadata (SI, HGS, ADF) ----------
    for col in ("target_SI", "HGS_pp_avg", "ADF_pp_avg"):
        if col in meta_te.columns:
            # Convert to string, strip % and whitespace, then coerce to numeric
            meta_te[col] = (
                meta_te[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            meta_te[col] = pd.to_numeric(meta_te[col], errors="coerce")

    # ---------- config (match your 10-fold grids) ----------
    TARGETS = [
        ("Splicing Index", "target_SI"),
        ("Hand Grip Strength (%)", "HGS_pp_avg"),
        ("Average Ankle Dorsiflexion (%)", "ADF_pp_avg"),
    ]
    PREPROCESS_GRID = [
        [], ["EMSC"], ["Normalization"], ["SNV"], ["Second Derivative"],
        ["EMSC","Normalization"], ["EMSC","SNV"], ["EMSC","Second Derivative"],
        ["Normalization","EMSC"], ["Normalization","SNV"], ["Normalization","Second Derivative"],
        ["SNV","EMSC"], ["SNV","Normalization"], ["SNV","Second Derivative"],
        ["Second Derivative","EMSC"], ["Second Derivative","Normalization"], ["Second Derivative","SNV"],
    ]
    PLS_COMPONENTS  = list(range(2, 19))
    IPLS_COMPONENTS = list(range(2, 11))
    IPLS_INTERVALS  = [5, 10, 15]

    def generate_random_rf_params(num_iterations=20, seed=42):
        rng = _random.Random(seed)
        grid = []
        for _ in range(num_iterations):
            grid.append({
                "n_estimators":      rng.choice([50, 100, 150, 160, 170, 180, 190]),
                "max_features":      "sqrt",
                "max_depth":         rng.choice([10, 15, 20, 25]),
                "min_samples_split": rng.choice([2, 10, 50, 100, 130, 140, 150, 160, 170, 180]),
                "min_samples_leaf":  rng.choice([1, 10, 30, 40, 47, 48, 50, 55, 60]),
            })
        return grid
    
    RF_GRID = generate_random_rf_params(20, seed=args.random_state)

    ACFNN_GRID = [
        {"hidden_layer_sizes": (256, 128),      "activation":"relu", "alpha":1e-4, "learning_rate_init":1e-3, "batch_size":64,  "max_iter":200},
        {"hidden_layer_sizes": (512, 256),      "activation":"relu", "alpha":1e-4, "learning_rate_init":5e-4, "batch_size":64,  "max_iter":300},
        {"hidden_layer_sizes": (256,256,128),   "activation":"relu", "alpha":1e-5, "learning_rate_init":1e-3, "batch_size":128, "max_iter":300},
    ]

    def _is_simpler(model_a, hp_a, model_b, hp_b):
        order = {"pls": 0, "ipls": 1, "rf": 2, "acfnn": 3}
        ma, mb = model_a.lower(), model_b.lower()
        if ma != mb:
            return order.get(ma, 99) < order.get(mb, 99)
        if ma == "pls":
            return int(hp_a["n_components"]) < int(hp_b["n_components"])
        if ma == "ipls":
            a = (int(hp_a["num_intervals"]), int(hp_a["n_components"]))
            b = (int(hp_b["num_intervals"]), int(hp_b["n_components"]))
            return a < b
        if ma == "rf":
            return int(hp_a["n_estimators"]) < int(hp_b["n_estimators"])
        if ma == "acfnn":
            ha = hp_a.get("hidden_layer_sizes", (128, 64))
            hb = hp_b.get("hidden_layer_sizes", (128, 64))
            return (sum(ha), len(ha)) < (sum(hb), len(hb))
        return False

    # ---------- core ----------
    rm_tr = GlobalGrouped(all_spectra=all_tr, meta=meta_tr, reps=REPS)
    Xte_full = all_te.T  # (reps*N_test, features)
    # ---------------------- Fold-making helpers (sample-level) ----------------------
    
    def _iter_sample_folds(
        n_samples: int,
        n_splits: int,
        seed: int,
        strat_labels: np.ndarray | None = None,
    ):
        """
        Yield (train_sample_ids, test_sample_ids) at the SAMPLE level.
        - If strat_labels provided and valid for stratification, use StratifiedKFold(shuffle=True).
        - Else use KFold(shuffle=True).
        """
        sample_ids = np.arange(n_samples, dtype=int)
    
        # Decide whether stratification is feasible (>= n_splits per class, ≥2 classes)
        use_strat = False
        if strat_labels is not None:
            labs = np.asarray(strat_labels)
            if labs.shape[0] == n_samples and len(np.unique(labs)) >= 2:
                counts = {c: (labs == c).sum() for c in np.unique(labs)}
                if min(counts.values()) >= n_splits:
                    use_strat = True
    
        if use_strat:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for tr, te in skf.split(sample_ids, strat_labels):
                yield sample_ids[tr], sample_ids[te]
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for tr, te in kf.split(sample_ids):
                yield sample_ids[tr], sample_ids[te]
    
    
    def _warn_fold_balance(
        fold_id: int,
        tr_ids: np.ndarray,
        te_ids: np.ndarray,
        strat_labels_full: np.ndarray | None,
        tol_frac: float = 0.20,    # 20% absolute deviation allowed
        min_test: int = 4,         # tiny fold warning
    ):
        """
        Emit light warnings about fold size and class balance drift (when labels present).
        """
        if te_ids.size < min_test:
            print(f"[WARN] outer fold {fold_id}: test fold very small (n={te_ids.size}).")
    
        if strat_labels_full is None:
            return
    
        labs = strat_labels_full
        glob = {c: (labs == c).mean() for c in np.unique(labs)}
        te = {c: (labs[te_ids] == c).mean() for c in np.unique(labs)}
    
        for c in glob:
            if abs(te[c] - glob[c]) > tol_frac:
                print(
                    f"[WARN] outer fold {fold_id}: class '{c}' proportion drift "
                    f"({te[c]:.2f} vs global {glob[c]:.2f})."
                )
    
    
    def _expand_sample_ids_to_spectra(mask_sample_ids: np.ndarray, reps: int) -> np.ndarray:
        """
        Convert a boolean mask or an array of sample IDs into a boolean mask for the
        (reps * N)-long spectra axis.
        """
        if mask_sample_ids.dtype == bool:
            return np.repeat(mask_sample_ids, reps)
        # it's an array of sample indices
        N = int(mask_sample_ids.max()) + 1 if mask_sample_ids.size else 0
        m = np.zeros(N, dtype=bool)
        m[mask_sample_ids] = True
        return np.repeat(m, reps)
    def _spectral_groups_from_mask(mask_bool: np.ndarray, reps: int, N: int) -> np.ndarray:
        """
        Build group ids aligned to X[mask_bool] row order.
        Each row corresponds to a spectrum; we map that back to its sample id and
        then remap sample ids to compact 0..(n_unique-1) in appearance order.
        """
        # sample id for every spectrum in the full design (0..N-1 repeated 'reps' times)
        spec_sample_ids_full = np.repeat(np.arange(N, dtype=int), reps)
        # restrict to the selected rows; this matches X[mask_bool]
        spec_sample_ids_sel = spec_sample_ids_full[mask_bool]
        # compact mapping
        _, inv = np.unique(spec_sample_ids_sel, return_inverse=True)
        return inv

    def nested_select_one(model_name: str, methods: list, target_col: str,
                          n_outer=10, n_inner=5, seed=42):
        """
        Group-aware nested CV on TRAIN to choose HP via 1-SE rule toward simplicity.
        Now uses sample-level shuffled (and optionally stratified) folds, expanded back
        to spectra so replicate grouping is preserved automatically.
    
        Returns: dict with selection, CV metrics, and per-fold chosen HPs.
        """
        # ---- Build TRAIN matrices at the current target ----
        X, Y, y_s, groups, meta_kept = rm_tr.build_xy(target_col, include_types=INCLUDE_TYPES)
        # meta_kept has exactly N rows (patients) and aligns with y_s
        N = len(meta_kept)
        reps_here = rm_tr.reps
        spec_sample_ids_full = np.repeat(np.arange(N), reps_here)

        # sample id (0..N-1) for every spectrum row in X (length = reps_here * N)
        spec_sample_ids_full = np.repeat(np.arange(N, dtype=int), reps_here)

    
        # Build stratification labels if controls are present
        # Label map: DM1 -> 1, Control -> 0
        strat_labels = None
        if "Type" in meta_kept.columns:
            if meta_kept["Type"].isin(["DM1", "Control"]).any():
                strat_labels = (meta_kept["Type"].to_numpy() == "DM1").astype(int)
    
        # ---- Define candidate HP lists per model ----
        if model_name == "pls":
            hp_candidates = [{"n_components": int(c)} for c in PLS_COMPONENTS]
        elif model_name == "ipls":
            hp_candidates = [{"n_components": int(c), "num_intervals": int(I)}
                             for c in IPLS_COMPONENTS for I in IPLS_INTERVALS]
        elif model_name == "rf":
            hp_candidates = RF_GRID
        elif model_name == "acfnn":
            hp_candidates = ACFNN_GRID
        else:
            raise ValueError(model_name)
    
        # ---- OUTER folds (sample-level; shuffled, optionally stratified) ----
        n_outer = min(n_outer, N)
        y_true_all, y_pred_all = [], []
        inner_trace = []
        fold_idx = 0
    
        for tr_s, te_s in _iter_sample_folds(
            n_samples=N, n_splits=n_outer, seed=seed, strat_labels=strat_labels
        ):
            fold_idx += 1
    
            # Fold health / balance warnings
            _warn_fold_balance(
                fold_id=fold_idx, tr_ids=tr_s, te_ids=te_s,
                strat_labels_full=strat_labels, tol_frac=0.20, min_test=4
            )
    
            # Expand to spectra-level masks (keeps 9 reps together)
            mask_tr = np.isin(np.repeat(np.arange(N), reps_here), tr_s)
            mask_te = np.isin(np.repeat(np.arange(N), reps_here), te_s)
    
            Xtr, Xte = X[mask_tr], X[mask_te]
            Ytr, Yte = Y[mask_tr], Y[mask_te]
    
            # ---- INNER folds on TRAIN (sample-level; shuffled, optionally stratified) ----
            n_inner_eff = min(n_inner, max(2, len(np.unique(tr_s))))
            strat_labels_tr = strat_labels[tr_s] if strat_labels is not None else None
    
            def score_hp(hp):
                mse_folds = []
                for itr_s, iva_s in _iter_sample_folds(
                    n_samples=len(tr_s),
                    n_splits=n_inner_eff,
                    seed=seed + 17,
                    strat_labels=strat_labels_tr
                ):
                    inner_tr_samples = tr_s[itr_s]
                    inner_va_samples = tr_s[iva_s]
            
                    m_tr = np.isin(np.repeat(np.arange(N), reps_here), inner_tr_samples)
                    m_va = np.isin(np.repeat(np.arange(N), reps_here), inner_va_samples)
            
                    Xitr, Xiva = X[m_tr], X[m_va]
                    Yitr, Yiva = Y[m_tr], Y[m_va]
            
                    if model_name == "ipls":
                        groups_tr_inner = spec_sample_ids_full[m_tr]
                        (a, b), _ = _best_interval_grouped(
                            X_tr=Xitr, Y_tr=Yitr,
                            groups_tr=groups_tr_inner,
                            preprocess_methods=methods,
                            n_components=int(hp["n_components"]),
                            num_intervals=int(hp["num_intervals"]),
                            reps=reps_here,
                            n_splits=max(3, min(5, len(inner_tr_samples)))
                        )
                        Xt_tr, Xt_va = _fit_transform_pair(Xitr[:, a:b], Xiva[:, a:b], methods)
                        mdl = PLSRegression(n_components=int(hp["n_components"]))
                        mdl.fit(Xt_tr, Yitr)
                        Yhat = mdl.predict(Xt_va)
            
                    else:
                        Xt_tr, Xt_va = _fit_transform_pair(Xitr, Xiva, methods)
                        if model_name == "pls":
                            mdl = PLSRegression(n_components=int(hp["n_components"]))
                            mdl.fit(Xt_tr, Yitr)
                            Yhat = mdl.predict(Xt_va)
            
                        elif model_name == "rf":
                            # 1 thread here to avoid nested parallel deadlocks
                            rf = RandomForestRegressor(**hp, random_state=seed, n_jobs=1)
                            rf.fit(Xt_tr, Yitr.ravel())
                            Yhat = rf.predict(Xt_va).reshape(-1, 1)
            
                        elif model_name == "acfnn":
                            # Torch cannot be inside multiprocessing on Windows
                            mlp = TorchRegressor(random_state=seed, **hp)
                            mlp.fit(Xt_tr, Yitr.ravel())
                            Yhat = mlp.predict(Xt_va).reshape(-1, 1)
            
                    m_true = _sample_means_stds(Yiva, reps=reps_here)[0]
                    m_pred = _sample_means_stds(Yhat, reps=reps_here)[0]
                    mse_folds.append(mean_squared_error(m_true, m_pred))
            
                return float(np.mean(mse_folds)), float(np.std(mse_folds))
            
            
            # ---- Parallel-safe HP evaluation ----
            if model_name == "acfnn":
                # NO joblib, run sequential to avoid worker crashes
                hp_stats = [score_hp(hp) for hp in hp_candidates]
            else:
                # Use thread-based parallelism to avoid Windows process crashes
                hp_stats = Parallel(n_jobs=4, backend="threading")(
                    delayed(score_hp)(hp) for hp in hp_candidates
                )

            mus = [m for (m, s) in hp_stats]
            best_ix = int(np.argmin(mus))
            mu_best, sd_best = hp_stats[best_ix]
            pick = hp_candidates[best_ix]
    
            # 1-SE toward simplicity
            for h, (m, s) in zip(hp_candidates, hp_stats):
                if m <= mu_best + sd_best + 1e-12:
                    if _is_simpler(model_name, h, model_name, pick):
                        pick = h
    
            inner_trace.append({
                "outer_fold": fold_idx,
                "selected_hp": pick,
                "mu_best": float(mu_best),
                "sd_best": float(sd_best),
            })
    
            # ---- Outer evaluation on this fold (train with pick → predict outer-test) ----
            # ---- Outer evaluation on this fold (train with pick → predict outer-test) ----
            if model_name == "ipls":
                groups_tr_outer = spec_sample_ids_full[mask_tr]
                (a, b), _ = _best_interval_grouped(
                    X_tr=Xtr, Y_tr=Ytr,
                    groups_tr=groups_tr_outer,
                    preprocess_methods=methods,
                    n_components=int(pick["n_components"]),
                    num_intervals=int(pick["num_intervals"]),
                    reps=reps_here,
                    n_splits=max(3, min(5, len(tr_s))),
                )
                Xt_tr, Xt_te = _fit_transform_pair(Xtr[:, a:b], Xte[:, a:b], methods)
                mdl = PLSRegression(n_components=int(pick["n_components"]))
                mdl.fit(Xt_tr, Ytr)
                Yhat = mdl.predict(Xt_te)

            else:
                Xt_tr, Xt_te = _fit_transform_pair(Xtr, Xte, methods)
                if model_name == "pls":
                    mdl = PLSRegression(n_components=int(pick["n_components"]))
                    mdl.fit(Xt_tr, Ytr); Yhat = mdl.predict(Xt_te)
                elif model_name == "rf":
                    rf = RandomForestRegressor(**pick, random_state=seed, n_jobs=-1)
                    rf.fit(Xt_tr, Ytr.ravel()); Yhat = rf.predict(Xt_te).reshape(-1, 1)
                elif model_name == "acfnn":
                    mlp = TorchRegressor(random_state=seed, **pick)
                    mlp.fit(Xt_tr, Ytr.ravel())
                    Yhat = mlp.predict(Xt_te).reshape(-1, 1)
    
            m_true = _sample_means_stds(Yte, reps=reps_here)[0]
            m_pred = _sample_means_stds(Yhat, reps=reps_here)[0]
            y_true_all.append(m_true); y_pred_all.append(m_pred)
    
        # ---- Aggregate CV metrics across outer folds ----
        YT = np.concatenate(y_true_all).reshape(-1, 1)
        YP = np.concatenate(y_pred_all).reshape(-1, 1)
        mets = _metrics(YT, YP)
    
        # ---- Choose modal HP across outer folds (ties broken downstream) ----
        from collections import Counter
        def _hp_key(hp):
            if model_name == "pls":   return f"pls|C={hp['n_components']}"
            if model_name == "rf":    return f"rf|{_rf_tag(hp)}"
            if model_name == "acfnn": return f"ac|{_ac_tag(hp)}"
            if model_name == "ipls":  return f"ipls|C={hp['n_components']}__I={hp['num_intervals']}"
            return str(hp)
    
        counts = Counter([_hp_key(r["selected_hp"]) for r in inner_trace])
        modal_key, _ = counts.most_common(1)[0]
        sel_hp = next(r["selected_hp"] for r in inner_trace if _hp_key(r["selected_hp"]) == modal_key)
    
        return {
            "model": model_name,
            "methods": methods,
            "selected_hp": sel_hp,
            "cv_metrics": mets,
            "inner_trace": inner_trace,
        }


    # For each target: do nested selection per (model, methods), pick the overall best by CV, then train on FULL TRAIN and eval on TEST
    for nice, tcol in TARGETS:
        print(f"\n=== External evaluation (train 1–2 -> test 3) for {nice} ({tcol}) ===")

        # Build TRAIN and TEST matrices for this target
        Xtr_all, Ytr_all, ytr_s, groups_tr_all, meta_tr_kept = rm_tr.build_xy(tcol, include_types=INCLUDE_TYPES)
        tr_keep_mask_samples = np.ones(len(meta_tr_kept), dtype=bool)  # already filtered in build_xy
        mask_tr_full = np.repeat(tr_keep_mask_samples, REPS)
        # TEST keep rows with numeric target
        te_raw = meta_te[tcol]
        te_num = pd.to_numeric(te_raw, errors="coerce")
        te_keep = te_num.notna().to_numpy()

        # Debug / safety checks
        n_total = len(te_keep)
        n_keep = int(te_keep.sum())
        print(f"[DEBUG] Test non-NaN for {tcol}: {n_keep} / {n_total}")
        if n_keep == 0:
            print("[DEBUG] meta_te[", tcol, "] values:")
            print(te_raw)
            raise ValueError(
                f"No finite test values found for target '{tcol}' after numeric coercion. "
                f"Check y_metadata_test CSV for that column."
            )

        Xte = Xte_full[np.repeat(te_keep, REPS)]
        yte = te_num[te_keep].to_numpy(dtype=float)


        tgt_out = os.path.join(args.out_dir, tcol)
        os.makedirs(tgt_out, exist_ok=True)

        # Accumulate per-(model,preproc) rows with CV & External metrics for the SELECTED HP
        rows = []
        best = None  # track best by CV rmse, then simplicity

        # iterate preprocessing chains
        for methods in PREPROCESS_GRID:
            mlabel = " + ".join(methods) if methods else "No Preprocessing"

            for model in ("pls", "rf", "acfnn", "ipls"):
                sel = nested_select_one(model, methods, tcol, n_outer=args.outer_folds, n_inner=args.inner_folds, seed=args.random_state)
                hp = sel["selected_hp"]
                cv_mets = sel["cv_metrics"]

                # Train on FULL TRAIN with selected HP; for iPLS reselect interval on full train (grouped)
                if model == "ipls":
                    (a, b), _ = _best_interval_grouped(
                        X_tr=Xtr_all, Y_tr=Ytr_all, groups_tr=groups_tr_all,
                        preprocess_methods=methods,
                        n_components=int(hp["n_components"]), num_intervals=int(hp["num_intervals"]),
                        reps=REPS, n_splits=max(3, min(5, int(len(np.unique(groups_tr_all))))),
                    )
                    Xt_tr, Xt_te = _fit_transform_pair(Xtr_all[:, a:b], Xte[:, a:b], methods)
                    mdl = PLSRegression(n_components=int(hp["n_components"]))
                    mdl.fit(Xt_tr, Ytr_all)
                    yhat_ext_spec = mdl.predict(Xt_te).ravel()
                    hp_tag = f"C={hp['n_components']}__I={hp['num_intervals']}"
                else:
                    Xt_tr, Xt_te = _fit_transform_pair(Xtr_all, Xte, methods)
                    if model == "pls":
                        mdl = PLSRegression(n_components=int(hp["n_components"]))
                        mdl.fit(Xt_tr, Ytr_all); yhat_ext_spec = mdl.predict(Xt_te).ravel()
                        hp_tag = f"C={hp['n_components']}"
                    elif model == "rf":
                        rf = RandomForestRegressor(**hp, random_state=args.random_state, n_jobs=-1)
                        rf.fit(Xt_tr, Ytr_all.ravel()); yhat_ext_spec = rf.predict(Xt_te)
                        hp_tag = _rf_tag(hp)
                    elif model == "acfnn":
                        mlp = TorchRegressor(random_state=args.random_state, **hp)
                        mlp.fit(Xt_tr, Ytr_all.ravel())
                        yhat_ext_spec = mlp.predict(Xt_te)
                        hp_tag = _ac_tag(hp)


                # Collapse predictions to sample level and compute EXTERNAL metrics
                y_pred_s, y_pred_std = _sample_means_stds(yhat_ext_spec, reps=REPS)
                ext_mets = _metrics(yte, y_pred_s)

                rows.append({
                    "target": tcol,
                    "model": model,
                    "preprocessing": mlabel,
                    "hp": hp_tag,
                    # CV metrics (selection basis)
                    "cv_rmse": cv_mets["rmse"], "cv_r2": cv_mets["r2"],
                    "cv_mae": cv_mets["mae"], "cv_medae": cv_mets["medae"], "cv_evs": cv_mets["evs"],
                    # External metrics (report only; not used for selection)
                    "ext_rmse": ext_mets["rmse"], "ext_r2": ext_mets["r2"],
                    "ext_mae": ext_mets["mae"], "ext_medae": ext_mets["medae"], "ext_evs": ext_mets["evs"],
                })

                # Track winner by CV RMSE (then simplicity)
                if (best is None) or (cv_mets["rmse"] < best["cv_rmse"] - 1e-12) or \
                   (abs(cv_mets["rmse"] - best["cv_rmse"]) <= 1e-12 and _is_simpler(model, hp, best["model"], best["hp"])):
                    best = {
                        "model": model, "methods": methods, "hp": hp, "hp_tag": hp_tag,
                        "cv_rmse": cv_mets["rmse"], "ext_preds_s": y_pred_s, "ext_preds_std": y_pred_std,
                    }

        # Save per-target consolidated CSV
        df_all = pd.DataFrame(rows).sort_values(by=["cv_rmse", "model", "preprocessing", "hp"]).reset_index(drop=True)
        csv_path = os.path.join(tgt_out, f"{tcol}__external_eval_SELECTED_HP_per_model.csv")
        df_all.to_csv(csv_path, index=False)
        print(f"[METRICS] wrote {csv_path}  ({len(df_all)} rows)")

        if best is None:
            print(f"[WARN] No winner computed for {nice}.")
            continue

        # Winner artifacts
        methods_label = " + ".join(best["methods"]) if best["methods"] else "No Preprocessing"
        methods_path  = _safe_methods_path(best["methods"])
        winner_dir = os.path.join(tgt_out, f"winner__{best['model']}__{methods_path}")
        os.makedirs(winner_dir, exist_ok=True)

        # Parity plot on EXTERNAL (with y-error bars from spectra->sample collapse)
        parity_plot_sample_level(
            y_true_sample=yte.reshape(-1,1),
            y_pred_sample=best["ext_preds_s"].reshape(-1,1),
            y_true_sample_std=None,
            y_pred_sample_std=best["ext_preds_std"].reshape(-1,1),
            target_names=(nice,),
            out_dir=winner_dir,
            fname_prefix="parity_external",
            title_suffix=f"{best['model'].upper()} | {methods_label} | HP: {best['hp_tag']}",
        )

        # Tidy predictions on EXTERNAL for the winner
        save_outer_predictions_excel(
            y_true_sample=yte.reshape(-1,1),
            y_pred_sample=best["ext_preds_s"].reshape(-1,1),
            meta_kept=None,
            target_order=(tcol,),
            out_path=os.path.join(winner_dir, f"{tcol}__winner_predictions_EXTERNAL.xlsx"),
            y_true_sample_std=None,
            y_pred_sample_std=best["ext_preds_std"].reshape(-1,1),
        )

        print(f"[BEST] {nice}: {best['model']} + {methods_label} [{best['hp_tag']}] — CV_RMSE={best['cv_rmse']:.4f}")
