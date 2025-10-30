# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 20:48:44 2025

@author: notha
"""

# -*- coding: utf-8 -*-
"""
Nested 10-Fold CV (group-aware) with 1-SE model selection.
Runs once per (model, preprocessing) and saves:
- selected.xlsx (outer test preds at sample level)
- selection_trace.csv (per outer fold: μ, σ, chosen HP; iPLS logs intervals)
- consolidated metrics sheet
- parity plot + tidy preds for overall winner (per target)

Compatible with your main.py launcher.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import os
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, median_absolute_error, explained_variance_score,
)

# local modules (no "modules." prefix so it matches your main.py layout)
from preprocessing import Preprocessing


# ====================== Utilities ======================

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

def _groups_for_spectra(n_samples: int, reps: int = 9) -> np.ndarray:
    return np.repeat(np.arange(n_samples, dtype=int), reps)

def _sample_means_stds(y_like: np.ndarray, reps: int = 9):
    """Return (means, stds) collapsing reps spectra per sample. Works with (9N,) or (9N,K)."""
    y = np.asarray(y_like)
    assert (y.shape[0] % reps) == 0, "length must be multiple of reps"
    N = y.shape[0] // reps
    if y.ndim == 1:
        y3 = y.reshape(N, reps, 1)
        return y3.mean(axis=1).ravel(), y3.std(axis=1).ravel()
    y3 = y.reshape(N, reps, -1)
    return y3.mean(axis=1), y3.std(axis=1)

def _sample_means(y_like: np.ndarray, reps: int = 9) -> np.ndarray:
    y = np.asarray(y_like)
    assert (y.shape[0] % reps) == 0
    N = y.shape[0] // reps
    if y.ndim == 1:
        return y.reshape(N, reps).mean(axis=1)
    return y.reshape(N, reps, -1).mean(axis=1)

def _rmse_r2_sample_level(y_true_s: np.ndarray, y_pred_s: np.ndarray) -> Tuple[float, float]:
    return float(np.sqrt(mean_squared_error(y_true_s, y_pred_s))), float(r2_score(y_true_s, y_pred_s))

def _make_intervals(n_features: int, num_intervals: int):
    step = n_features // num_intervals
    return [(i*step, (i+1)*step if i < num_intervals-1 else n_features) for i in range(num_intervals)]

def _safe_methods_path(methods):
    return "+".join(m.replace(" ", "_") for m in (methods or [])) if methods else "none"

def _rf_tag(hp: dict) -> str:
    return (f"ne{hp['n_estimators']}_md{hp['max_depth']}_"
            f"mss{hp['min_samples_split']}_msl{hp['min_samples_leaf']}_mf{hp['max_features']}")

def _ac_tag(hp: dict) -> str:
    hls = "-".join(map(str, hp["hidden_layer_sizes"])) if isinstance(hp["hidden_layer_sizes"], (list, tuple)) else str(hp["hidden_layer_sizes"])
    return (f"hls{hls}_act{hp.get('activation','relu')}_bs{hp.get('batch_size','NA')}"
            f"_lr{hp.get('learning_rate_init','NA')}_wd{hp.get('alpha','0')}_mi{hp.get('max_iter','NA')}")

def _pls_tag(hp: dict) -> str:
    return f"C={int(hp['n_components'])}"

def _ipls_tag(hp: dict) -> str:
    return f"C={int(hp['n_components'])}__I={int(hp['num_intervals'])}"

def _hp_tag(model: str, hp: dict) -> str:
    m = model.lower()
    if m == "pls": return _pls_tag(hp)
    if m == "ipls": return _ipls_tag(hp)
    if m == "rf": return _rf_tag(hp)
    if m == "acfnn": return _ac_tag(hp)
    return str(hp)

def _is_simpler(model_a, hp_a, model_b, hp_b):
    """Simplicity order used for 1-SE tie-breaking."""
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

class TorchRegressor:
    """
    Simple feedforward regressor with GPU support.
    Expects a single target column.
    API subset: .fit(X, y), .predict(X) returning shape (n,1)
    """

    def __init__(self,
                 hidden_layer_sizes=(256,128),
                 activation="relu",
                 learning_rate_init=1e-3,
                 alpha=1e-4,           # L2 weight decay
                 max_iter=200,
                 batch_size=64,
                 random_state=42):
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

        optimizer = optim.Adam(self.model_.parameters(),
                               lr=self.learning_rate_init,
                               weight_decay=self.alpha)
        loss_fn = nn.MSELoss()

        # minibatch training
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


# ====================== iPLS inner interval search (grouped) ======================

def _best_interval_grouped(
    X_tr: np.ndarray, Y_tr: np.ndarray, groups_tr: np.ndarray,
    preprocess_methods, n_components: int, num_intervals: int,
    reps: int, n_splits: int = 3, random_state: int = 42
):
    """
    Choose best contiguous interval via GroupKFold on the (train-only) data.
    Returns (best_interval (a,b), best_mse).
    """
    intervals = _make_intervals(X_tr.shape[1], num_intervals)
    n_splits = max(2, min(n_splits, int(len(np.unique(groups_tr)))))
    inner = GroupKFold(n_splits=n_splits)
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
            elif m in ("", "No Preprocessing", "No preprocessing", "no preprocessing"):
                pass
            else:
                raise ValueError(f"Unknown preprocessing: {m}")
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


# ====================== Core class ======================

@dataclass
class GlobalGroupedCV:
    all_spectra: np.ndarray  # (features, 9N) in your loaders; transposed to (9N, features) here
    meta: pd.DataFrame       # N rows (samples)
    reps: int = 9

    def _build_xy(self, target_cols: List[str], include_types=("DM1", "Control")):
        meta0 = self.meta[self.meta["Type"].isin(include_types)].copy().reset_index(drop=True)
        # Study-specific defaulting for controls:
        meta0.loc[meta0["Type"] == "Control", "target_SI"]  = 0
        meta0.loc[meta0["Type"] == "Control", "HGS_pp_avg"] = 100
        meta0.loc[meta0["Type"] == "Control", "ADF_pp_avg"] = 100

        meta0 = meta0.dropna(subset=list(target_cols)).reset_index(drop=True)
        N = len(meta0)

        keep_cols = np.concatenate([np.arange(i*self.reps, (i+1)*self.reps) for i in range(N)], dtype=int)
        X_all = self.all_spectra[:, keep_cols].T  # (9N, features)

        Y_s = meta0[list(target_cols)].to_numpy(dtype=float)  # (N, K)
        Y   = np.repeat(Y_s, self.reps, axis=0)               # (9N, K)
        groups = _groups_for_spectra(N, reps=self.reps)       # (9N,)

        return X_all, Y, Y_s, groups, meta0

    def _fit_transform(self, X_train: np.ndarray, X_apply: np.ndarray, methods: list):
        """Train-only preprocessing, apply to validation/test (no leakage)."""
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
            elif m in ("", "No Preprocessing", "No preprocessing", "no preprocessing"):
                pass
            else:
                raise ValueError(f"Unknown preprocessing: {m}")
        return Xtr.T.astype(np.float32), Xap.T.astype(np.float32)


# ====================== CLI runner (Nested CV, Group-aware) ======================

if __name__ == "__main__":
    import argparse
    from data_loader import load_data
    from plot_parity import parity_plot_sample_level, save_outer_predictions_excel
    import random as _random
    from collections import Counter

    # ---------- parse CLI ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--meta_path", required=True)
    ap.add_argument("--include_types", default="DM1,Control")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--outer_folds", type=int, default=10)
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    INCLUDE_TYPES = tuple([s.strip() for s in (args.include_types or "").split(",") if s.strip()])
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- load data ----------
    wavenumbers, averaged_spectra, all_spectra, meta = load_data(
        data_dir=args.data_dir,
        metadata_path=args.meta_path,
        include_types=INCLUDE_TYPES,
        return_filenames=False,
        strict=False,
        report_samples=5,
    )

    # ---------- config (matches your old main grids) ----------
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

    RUN_PLS = True
    RUN_RF = True
    RUN_IPLS = True
    RUN_ACFNN = True

    # ---------- core ----------
    rm = GlobalGroupedCV(all_spectra=all_spectra, meta=meta, reps=9)

    def _select_and_eval(model_name: str, methods: List[str], hp_candidates: List[Any],
                         target_col: str, n_outer=10, n_inner=5, seed=42):
        """
        Runs nested GroupKFold once for (model, methods) on the given target.
        Inner CV: choose HP per outer fold via 1-SE toward simplicity.
        Outer test: fit once on outer-train with chosen HP and predict outer-test.
        Returns predictions at sample level, selection trace, summary HP stats, and metrics.
        """
        X, Y, Y_s, _, meta_kept = rm._build_xy([target_col], include_types=INCLUDE_TYPES)

        N = len(meta_kept)
        if N < 4:
            raise ValueError(f"Too few samples after filtering for target={target_col} (N={N}).")

        # ----- build sample-level ids and labels for outer split -----
        sample_ids = np.arange(N)                                  # (N,)
        spec_groups = np.repeat(sample_ids, rm.reps)               # (9N,) group label per spectrum (0..N-1)
        y_strat = (meta_kept["Type"] == "Control").astype(int).to_numpy()  # (N,) 0=DM1, 1=Control

        n_outer_eff = max(2, min(n_outer, int(len(sample_ids))))
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            outer_cv = StratifiedGroupKFold(n_splits=n_outer_eff, shuffle=True, random_state=seed)
            outer_splits = outer_cv.split(sample_ids, y_strat, groups=sample_ids)
        except ImportError:
            # Fallback: grouped CV with shuffled sample order (not stratified)
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(sample_ids)
            from sklearn.model_selection import GroupKFold
            outer_cv = GroupKFold(n_splits=n_outer_eff)
            outer_splits = outer_cv.split(shuffled, groups=shuffled)

        y_true_all, y_pred_all = [], []
        y_true_std_all, y_pred_std_all = [], []
        inner_trace = []
        chosen_hp_tags = []

        fold_idx = 0
        # --- sanity check: class balance & grouping ---
        meta_types = meta_kept["Type"].to_numpy()
        print(f"[CHECK] N samples for this target: {len(sample_ids)}; Controls={np.sum(y_strat==1)}, DM1={np.sum(y_strat==0)}")
        
        tmp_splits = list(outer_splits)  # materialize generator once
        for i, (tr_idx, te_idx) in enumerate(tmp_splits, 1):
            n_tr_ctl = int(np.sum(y_strat[tr_idx] == 1))
            n_tr_dm1 = int(np.sum(y_strat[tr_idx] == 0))
            n_te_ctl = int(np.sum(y_strat[te_idx] == 1))
            n_te_dm1 = int(np.sum(y_strat[te_idx] == 0))
        
            # verify replicate integrity (each sample contributes exactly rm.reps spectra)
            tr_s = sample_ids[tr_idx]
            te_s = sample_ids[te_idx]
            spec_groups = np.repeat(sample_ids, rm.reps)
            n_tr_specs = int(np.sum(np.isin(spec_groups, tr_s)))
            n_te_specs = int(np.sum(np.isin(spec_groups, te_s)))
        
            print(f"[FOLD {i:02d}] Train: DM1={n_tr_dm1}, Ctrl={n_tr_ctl} | Test: DM1={n_te_dm1}, Ctrl={n_te_ctl} "
                  f"| specs/train={n_tr_specs}, specs/test={n_te_specs} (should be multiples of {rm.reps})")
        
        # reuse the realized list for the main loop:
        outer_splits = tmp_splits

        for tr_s_idx, te_s_idx in outer_splits:
            fold_idx += 1
            tr_s = sample_ids[tr_s_idx]   # sample ids in train fold
            te_s = sample_ids[te_s_idx]   # sample ids in test  fold

            # ---- map sample-level split to spectrum-level masks (keeps all 9 spectra together) ----
            mask_tr = np.isin(spec_groups, tr_s)   # (9N,) True for spectra belonging to train samples
            mask_te = np.isin(spec_groups, te_s)   # (9N,) True for spectra belonging to test  samples

            Xtr, Xte = X[mask_tr], X[mask_te]
            Ytr, Yte = Y[mask_tr], Y[mask_te]
            groups_tr = spec_groups[mask_tr]       # (9N_tr,) replicate-aware group labels for inner CV


            # inner grouped CV for HP selection
            n_inner_eff = max(2, min(n_inner, int(len(np.unique(groups_tr)))))
            gkf_inner = GroupKFold(n_splits=n_inner_eff)

            def score_hp(hp):
                mse_list = []
                for itr, iva in gkf_inner.split(Xtr, groups=groups_tr):
                    Xitr, Xiva = Xtr[itr], Xtr[iva]
                    Yitr, Yiva = Ytr[itr], Ytr[iva]
                    groups_itr = groups_tr[itr]

                    if model_name == "ipls":
                        (a, b), _ = _best_interval_grouped(
                            X_tr=Xitr, Y_tr=Yitr, groups_tr=groups_itr,
                            preprocess_methods=methods,
                            n_components=int(hp["n_components"]),
                            num_intervals=int(hp["num_intervals"]),
                            reps=rm.reps, n_splits=min(5, n_inner_eff)
                        )
                        Xt_tr, Xt_va = rm._fit_transform(Xitr[:, a:b], Xiva[:, a:b], methods)
                        mdl = PLSRegression(n_components=int(hp["n_components"]))
                        mdl.fit(Xt_tr, Yitr)
                        Yhat = mdl.predict(Xt_va)
                    else:
                        Xt_tr, Xt_va = rm._fit_transform(Xitr, Xiva, methods)
                        if model_name == "pls":
                            mdl = PLSRegression(n_components=int(hp["n_components"]))
                            mdl.fit(Xt_tr, Yitr); Yhat = mdl.predict(Xt_va)
                        elif model_name == "rf":
                            rf = RandomForestRegressor(**hp, random_state=seed, n_jobs=-1)
                            rf.fit(Xt_tr, Yitr.ravel()); Yhat = rf.predict(Xt_va).reshape(-1, 1)
                        elif model_name == "acfnn":
                            mlp = TorchRegressor(random_state=seed, **hp)
                            mlp.fit(Xt_tr, Yitr.ravel())
                            Yhat = mlp.predict(Xt_va).reshape(-1, 1)

                        else:
                            raise ValueError(model_name)

                    m_true, _ = _sample_means_stds(Yiva, reps=rm.reps)
                    m_pred, _ = _sample_means_stds(Yhat, reps=rm.reps)
                    mse_list.append(mean_squared_error(m_true, m_pred))
                return float(np.mean(mse_list)), float(np.std(mse_list))

            hp_stats = Parallel(n_jobs=-1, backend="loky")(
                delayed(score_hp)(hp) for hp in hp_candidates
            )

            mus = [m for (m, s) in hp_stats]
            best_ix = int(np.argmin(mus))
            mu_best, sd_best = hp_stats[best_ix]
            pick = hp_candidates[best_ix]

            # 1-SE rule toward simplicity
            for h, (m, s) in zip(hp_candidates, hp_stats):
                if m <= mu_best + sd_best + 1e-12:
                    if _is_simpler(model_name, h, model_name, pick):
                        pick = h

            chosen_hp_tags.append(_hp_tag(model_name, pick))
            trace_row = {
                "outer_fold": fold_idx,
                "selected_hp": json.dumps(pick),
                "selected_hp_tag": _hp_tag(model_name, pick),
                "mu_best": float(mu_best),
                "sd_best": float(sd_best),
                "n_inner": int(n_inner_eff),
            }

            # Fit on full outer-train with chosen HP; for iPLS reselect interval on full outer-train
            if model_name == "ipls":
                (a_eval, b_eval), _ = _best_interval_grouped(
                    X_tr=Xtr, Y_tr=Ytr, groups_tr=groups_tr,
                    preprocess_methods=methods,
                    n_components=int(pick["n_components"]), num_intervals=int(pick["num_intervals"]),
                    reps=rm.reps, n_splits=min(5, n_inner_eff)
                )
                Xt_tr, Xt_te = rm._fit_transform(Xtr[:, a_eval:b_eval], Xte[:, a_eval:b_eval], methods)
                mdl = PLSRegression(n_components=int(pick["n_components"]))
                mdl.fit(Xt_tr, Ytr)
                Yhat = mdl.predict(Xt_te)
                trace_row.update({"interval_a": int(a_eval), "interval_b": int(b_eval)})
            else:
                Xt_tr, Xt_te = rm._fit_transform(Xtr, Xte, methods)
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


            inner_trace.append(trace_row)

            m_true, s_true = _sample_means_stds(Yte, reps=rm.reps)
            m_pred, s_pred = _sample_means_stds(Yhat, reps=rm.reps)
            y_true_all.append(m_true);    y_pred_all.append(m_pred)
            y_true_std_all.append(s_true); y_pred_std_all.append(s_pred)

        # Concatenate across outer folds
        YT = np.concatenate(y_true_all).reshape(-1, 1)
        YP = np.concatenate(y_pred_all).reshape(-1, 1)
        ST = np.concatenate(y_true_std_all).reshape(-1, 1)
        SP = np.concatenate(y_pred_std_all).reshape(-1, 1)

        # Modal HP tag (for reporting)
        modal_hp_tag = None
        if chosen_hp_tags:
            cnt = Counter(chosen_hp_tags)
            modal_hp_tag = cnt.most_common(1)[0][0]

        return {
            "y_true": YT, "y_pred": YP,
            "y_true_std": ST, "y_pred_std": SP,
            "metrics": _metrics(YT, YP),
            "inner_trace": inner_trace,
            "modal_hp_tag": modal_hp_tag,
            "n_outer": int(n_outer_eff)
        }

    # ---------- per-target run & outputs ----------
    for nice, tcol in TARGETS:
        tgt_root = os.path.join(args.out_dir, tcol)
        os.makedirs(tgt_root, exist_ok=True)
        pred_root = os.path.join(tgt_root, "all_combos_nestedcv")
        os.makedirs(pred_root, exist_ok=True)

        rows = []
        best = None  # {"rmse":..., "model":..., "methods":[...], "res":..., "hp_note":...}

        for methods in PREPROCESS_GRID:
            mlabel = " + ".join(methods) if methods else "No Preprocessing"
            mpath  = _safe_methods_path(methods)

            # ---- PLS (grid tuned inside) ----
            if RUN_PLS:
                model = "pls"
                model_dir = os.path.join(pred_root, f"{model}__{mpath}"); os.makedirs(model_dir, exist_ok=True)
                hp_candidates = [{"n_components": int(c)} for c in PLS_COMPONENTS]
                res = _select_and_eval(model, methods, hp_candidates, tcol,
                                       n_outer=args.outer_folds, n_inner=args.inner_folds, seed=args.random_state)

                # save predictions
                pd.DataFrame({f"{tcol}__true": res["y_true"].ravel(),
                              f"{tcol}__pred": res["y_pred"].ravel()}).to_excel(
                    os.path.join(model_dir, "selected.xlsx"), index=False
                )
                # save trace
                pd.DataFrame(res["inner_trace"]).to_csv(os.path.join(model_dir, "selection_trace.csv"), index=False)

                rows.append({
                    "target": tcol, "model": model, "preprocessing": mlabel,
                    "hp": "selected (nested)", "hp_modal_tag": res["modal_hp_tag"] or "",
                    "rmse": res["metrics"]["rmse"], "r2": res["metrics"]["r2"],
                    "mae": res["metrics"]["mae"], "medae": res["metrics"]["medae"], "evs": res["metrics"]["evs"],
                    "cv_method": "Nested GroupKFold", "n_outer": res["n_outer"], "n_inner": args.inner_folds,
                    "random_state": args.random_state
                })
                if (best is None) or (res["metrics"]["rmse"] < best["rmse"] - 1e-12) or \
                   (abs(res["metrics"]["rmse"] - best["rmse"]) <= 1e-12 and _is_simpler(model, {"n_components": 1}, best["model"], {"n_components": 9999})):
                    best = {"rmse": res["metrics"]["rmse"], "model": model, "methods": methods, "res": res, "hp_note": res["modal_hp_tag"]}

            # ---- RF ----
            if RUN_RF:
                model = "rf"
                model_dir = os.path.join(pred_root, f"{model}__{mpath}"); os.makedirs(model_dir, exist_ok=True)
                res = _select_and_eval(model, methods, RF_GRID, tcol,
                                       n_outer=args.outer_folds, n_inner=args.inner_folds, seed=args.random_state)
                pd.DataFrame({f"{tcol}__true": res["y_true"].ravel(),
                              f"{tcol}__pred": res["y_pred"].ravel()}).to_excel(
                    os.path.join(model_dir, "selected.xlsx"), index=False
                )
                pd.DataFrame(res["inner_trace"]).to_csv(os.path.join(model_dir, "selection_trace.csv"), index=False)

                rows.append({
                    "target": tcol, "model": model, "preprocessing": mlabel,
                    "hp": "selected (nested)", "hp_modal_tag": res["modal_hp_tag"] or "",
                    "rmse": res["metrics"]["rmse"], "r2": res["metrics"]["r2"],
                    "mae": res["metrics"]["mae"], "medae": res["metrics"]["medae"], "evs": res["metrics"]["evs"],
                    "cv_method": "Nested GroupKFold", "n_outer": res["n_outer"], "n_inner": args.inner_folds,
                    "random_state": args.random_state
                })
                if (best is None) or (res["metrics"]["rmse"] < best["rmse"] - 1e-12) or \
                   (abs(res["metrics"]["rmse"] - best["rmse"]) <= 1e-12 and _is_simpler(model, {"n_estimators": 1}, best["model"], {"n_estimators": 10**9})):
                    best = {"rmse": res["metrics"]["rmse"], "model": model, "methods": methods, "res": res, "hp_note": res["modal_hp_tag"]}

            # ---- AC-FNN ----
            if RUN_ACFNN:
                model = "acfnn"
                model_dir = os.path.join(pred_root, f"{model}__{mpath}"); os.makedirs(model_dir, exist_ok=True)
                res = _select_and_eval(model, methods, ACFNN_GRID, tcol,
                                       n_outer=args.outer_folds, n_inner=args.inner_folds, seed=args.random_state)
                pd.DataFrame({f"{tcol}__true": res["y_true"].ravel(),
                              f"{tcol}__pred": res["y_pred"].ravel()}).to_excel(
                    os.path.join(model_dir, "selected.xlsx"), index=False
                )
                pd.DataFrame(res["inner_trace"]).to_csv(os.path.join(model_dir, "selection_trace.csv"), index=False)

                rows.append({
                    "target": tcol, "model": model, "preprocessing": mlabel,
                    "hp": "selected (nested)", "hp_modal_tag": res["modal_hp_tag"] or "",
                    "rmse": res["metrics"]["rmse"], "r2": res["metrics"]["r2"],
                    "mae": res["metrics"]["mae"], "medae": res["metrics"]["medae"], "evs": res["metrics"]["evs"],
                    "cv_method": "Nested GroupKFold", "n_outer": res["n_outer"], "n_inner": args.inner_folds,
                    "random_state": args.random_state
                })
                # Simplicity: AC-FNN is last in hierarchy; only wins on strictly better RMSE
                if (best is None) or (res["metrics"]["rmse"] < best["rmse"] - 1e-12):
                    best = {"rmse": res["metrics"]["rmse"], "model": model, "methods": methods, "res": res, "hp_note": res["modal_hp_tag"]}

            # ---- iPLS ----
            if RUN_IPLS:
                model = "ipls"
                model_dir = os.path.join(pred_root, f"{model}__{mpath}"); os.makedirs(model_dir, exist_ok=True)
                hp_candidates = [{"n_components": int(c), "num_intervals": int(I)} for c in IPLS_COMPONENTS for I in IPLS_INTERVALS]
                res = _select_and_eval(model, methods, hp_candidates, tcol,
                                       n_outer=args.outer_folds, n_inner=args.inner_folds, seed=args.random_state)
                pd.DataFrame({f"{tcol}__true": res["y_true"].ravel(),
                              f"{tcol}__pred": res["y_pred"].ravel()}).to_excel(
                    os.path.join(model_dir, "selected.xlsx"), index=False
                )
                pd.DataFrame(res["inner_trace"]).to_csv(os.path.join(model_dir, "selection_trace.csv"), index=False)

                rows.append({
                    "target": tcol, "model": model, "preprocessing": mlabel,
                    "hp": "selected (nested)", "hp_modal_tag": res["modal_hp_tag"] or "",
                    "rmse": res["metrics"]["rmse"], "r2": res["metrics"]["r2"],
                    "mae": res["metrics"]["mae"], "medae": res["metrics"]["medae"], "evs": res["metrics"]["evs"],
                    "cv_method": "Nested GroupKFold", "n_outer": res["n_outer"], "n_inner": args.inner_folds,
                    "random_state": args.random_state
                })
                if (best is None) or (res["metrics"]["rmse"] < best["rmse"] - 1e-12) or \
                   (abs(res["metrics"]["rmse"] - best["rmse"]) <= 1e-12 and _is_simpler(model, {"n_components":1,"num_intervals":1},
                                                                                        best["model"], {"n_components":9999,"num_intervals":9999})):
                    best = {"rmse": res["metrics"]["rmse"], "model": model, "methods": methods, "res": res, "hp_note": res["modal_hp_tag"]}

        # consolidated metrics per target
        if rows:
            metrics_df = pd.DataFrame(rows).sort_values(by=["rmse", "model", "preprocessing"]).reset_index(drop=True)
        else:
            metrics_df = pd.DataFrame(columns=[
                "target", "model", "preprocessing", "hp", "hp_modal_tag",
                "rmse", "r2", "mae", "medae", "evs",
                "cv_method", "n_outer", "n_inner", "random_state"
            ])
        metrics_path = os.path.join(tgt_root, f"{tcol}__metrics_all_combos_NESTED_CV.xlsx")
        metrics_df.to_excel(metrics_path, index=False)
        print(f"[METRICS] wrote {metrics_path}  ({len(metrics_df)} rows)")

        # winner artifacts
        if best is None:
            print(f"[WARN] No models enabled for {nice}. Skipping winner artifacts.")
            continue

        methods_label = " + ".join(best["methods"]) if best["methods"] else "No Preprocessing"
        methods_path  = _safe_methods_path(best["methods"])
        winner_dir = os.path.join(tgt_root, f"winner__{best['model']}__{methods_path}")
        os.makedirs(winner_dir, exist_ok=True)

        # parity (with error bars from spectra→sample collapse)
        parity_plot_sample_level(
            y_true_sample=best["res"]["y_true"],
            y_pred_sample=best["res"]["y_pred"],
            y_true_sample_std=best["res"]["y_true_std"],
            y_pred_sample_std=best["res"]["y_pred_std"],
            target_names=(nice,),
            out_dir=winner_dir,
            fname_prefix="parity_nestedcv",
            title_suffix=f"{best['model'].upper()} | {methods_label} | HP: selected (nested){' | modal='+str(best['hp_note']) if best['hp_note'] else ''}"
        )

        # tidy predictions for the winner (this target only)
        from plot_parity import save_outer_predictions_excel  # already imported above; just for clarity
        save_outer_predictions_excel(
            y_true_sample=best["res"]["y_true"],
            y_pred_sample=best["res"]["y_pred"],
            meta_kept=None,
            target_order=(tcol,),
            out_path=os.path.join(winner_dir, f"{tcol}__winner_predictions_NESTED_CV.xlsx"),
            y_true_sample_std=best["res"]["y_true_std"],
            y_pred_sample_std=best["res"]["y_pred_std"],
        )

        print(f"[BEST] {nice}: {best['model']} + {methods_label} — RMSE={best['rmse']:.4f} (selected via nested GroupKFold)")
