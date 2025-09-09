# -*- coding: utf-8 -*-
"""
SPXY evaluation using best per-target models from the global 10-fold run.
This file is placed inside the 'modules/' folder, so we add the project root
to sys.path before importing from 'modules'.
"""

import os
import sys

# --- ensure project root is on sys.path when running from modules/ ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- stdlib / third-party ---
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

# --- your modules (now resolvable thanks to the path shim) ---
from modules.data_loader import load_data
from modules.preprocessing import Preprocessing
from modules.plot_parity import parity_plot_sample_level, save_outer_predictions_excel
from modules.ten_fold_cv_regression_global import _best_interval_grouped

# --- sklearn ---
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    median_absolute_error,
    explained_variance_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------- CONFIG ----------------------

TRAIN_DATA_DIR  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\data"
TRAIN_META_PATH = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata.csv"

GLOBAL_RESULTS_DIR = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Matthews_code_Global\results_global_hp_per_target"

OUT_DIR_SPXY = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\spxy_eval"
os.makedirs(OUT_DIR_SPXY, exist_ok=True)

TARGETS = [
    ("Splicing Index", "target_SI"),
    ("Hand Grip Strength (%)", "HGS_pp_avg"),
    ("Average Ankle Dorsiflexion (%)", "ADF_pp_avg"),
]

REPS = 9             # spectra per sample
CAL_FRAC = 0.80      # SPXY calibration fraction
ALPHA = 0.5          # SPXY weighting between X and y distances
N_REPEATS = 1        # set >1 to repeat with different SPXY starts


# ---------------------- HELPERS ----------------------

def _groups_for_spectra(n_samples: int, reps: int = 9) -> np.ndarray:
    return np.repeat(np.arange(n_samples, dtype=int), reps)

def _sample_means_stds(y_like: np.ndarray, reps: int = 9):
    y = np.asarray(y_like)
    assert (y.shape[0] % reps) == 0
    N = y.shape[0] // reps
    if y.ndim == 1:
        y3 = y.reshape(N, reps, 1)
        m  = y3.mean(axis=1).ravel()
        s  = y3.std(axis=1).ravel()
        return m, s
    y3 = y.reshape(N, reps, -1)
    return y3.mean(axis=1), y3.std(axis=1)

def _parse_methods(label: str) -> list[str]:
    if not label or label.lower().startswith("no"):
        return []
    return [p.strip() for p in label.split("+")]

def _parse_hp(model: str, hp_str: str):
    model = model.lower()
    if model == "pls":
        m = re.match(r"C=(\d+)", hp_str.strip());  assert m, f"Bad PLS hp: {hp_str}"
        return {"n_components": int(m.group(1))}
    if model == "ipls":
        m = re.match(r"C=(\d+)__I=(\d+)", hp_str.strip());  assert m, f"Bad iPLS hp: {hp_str}"
        return {"n_components": int(m.group(1)), "num_intervals": int(m.group(2))}
    if model == "rf":
        patt = r"ne(\d+)_md(\d+)_mss(\d+)_msl(\d+)_mf(\w+)"
        m = re.match(patt, hp_str.strip());        assert m, f"Bad RF hp: {hp_str}"
        ne, md, mss, msl, mf = m.groups()
        return {"n_estimators": int(ne), "max_depth": int(md),
                "min_samples_split": int(mss), "min_samples_leaf": int(msl),
                "max_features": "sqrt" if mf.lower()=="sqrt" else mf}
    if model == "acfnn":
        d = {}
        m = re.search(r"hls([0-9\-]+)", hp_str);   d["hidden_layer_sizes"] = tuple(int(x) for x in m.group(1).split("-")) if m else (128,64)
        m = re.search(r"act([a-zA-Z]+)", hp_str);  d["activation"] = (m.group(1) if m else "relu")
        m = re.search(r"bs(\d+)", hp_str);         d["batch_size"] = int(m.group(1)) if m else 64
        m = re.search(r"lr([0-9.]+)", hp_str);     d["learning_rate_init"] = float(m.group(1)) if m else 1e-3
        m = re.search(r"wd([0-9.]+)", hp_str);     d["alpha"] = float(m.group(1)) if m else 1e-4
        m = re.search(r"mi(\d+)", hp_str);         d["max_iter"] = int(m.group(1)) if m else 300
        return d
    raise ValueError(f"Unknown model: {model}")

def _fit_transform_pair(X_tr: np.ndarray, X_ap: np.ndarray, methods: list[str]):
    # scikit rows = samples; your Preprocessing expects (features, n)
    A = X_tr.T.copy(); B = X_ap.T.copy()
    for m in (methods or []):
        if m == "EMSC":
            ref = np.mean(A, axis=1)
            A = Preprocessing(A).emsc(A, reference=ref)
            B = Preprocessing(B).emsc(B, reference=ref)
        elif m == "Normalization":
            A = Preprocessing(A).normalize_spectrum(A)
            B = Preprocessing(B).normalize_spectrum(B)
        elif m == "SNV":
            A = Preprocessing(A).snv(A)
            B = Preprocessing(B).snv(B)
        elif m == "Second Derivative":
            A = Preprocessing(A).second_derivative(A)
            B = Preprocessing(B).second_derivative(B)
        elif m in ("No Preprocessing","No preprocessing","no preprocessing",""):
            pass
        else:
            raise ValueError(f"Unknown preprocessing: {m}")
    return A.T.astype(np.float32), B.T.astype(np.float32)

def _train_predict(model_name, methods, hp, Xcal, ycal, Xval, groups_cal):
    model_name = model_name.lower()

    if model_name == "pls":
        Xt_tr, Xt_va = _fit_transform_pair(Xcal, Xval, methods)
        mdl = PLSRegression(n_components=int(hp["n_components"]))
        mdl.fit(Xt_tr, ycal)
        return mdl.predict(Xt_va).ravel()

    if model_name == "rf":
        Xt_tr, Xt_va = _fit_transform_pair(Xcal, Xval, methods)
        rf = RandomForestRegressor(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            min_samples_split=hp["min_samples_split"], min_samples_leaf=hp["min_samples_leaf"],
            max_features=hp["max_features"], random_state=42, n_jobs=-1
        )
        rf.fit(Xt_tr, ycal.ravel())
        return rf.predict(Xt_va)

    if model_name == "acfnn":
        Xt_tr, Xt_va = _fit_transform_pair(Xcal, Xval, methods)
        mlp = MLPRegressor(random_state=42, **hp)
        mlp.fit(Xt_tr, ycal.ravel())
        return mlp.predict(Xt_va)

    if model_name == "ipls":
        num_intervals = int(hp["num_intervals"]); n_components = int(hp["n_components"])
        (a,b), _ = _best_interval_grouped(
            X_tr=Xcal, Y_tr=ycal, groups_tr=groups_cal,
            preprocess_methods=methods, n_components=n_components, num_intervals=num_intervals,
            reps=REPS, n_splits=max(3, min(5, int(len(np.unique(groups_cal))))),
            random_state=42
        )
        Xt_tr, Xt_va = _fit_transform_pair(Xcal[:, a:b], Xval[:, a:b], methods)
        mdl = PLSRegression(n_components=n_components)
        mdl.fit(Xt_tr, ycal)
        return mdl.predict(Xt_va).ravel()

    raise ValueError(f"Unsupported model: {model_name}")

def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "evs": float(explained_variance_score(y_true, y_pred)),
    }

def _spxy_split(Xsamp, ysamp, cal_frac=0.8, alpha=0.5, seed=None):
    """
    SPXY at the SAMPLE level using averaged spectra per sample.
    Returns (cal_idx, val_idx) arrays of sample indices.
    """
    rng = np.random.default_rng(seed)
    N = Xsamp.shape[0]
    Ncal = max(2, int(np.floor(cal_frac * N)))
    Nval = N - Ncal

    # Scale features for distance fairness
    Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(Xsamp)
    yz = StandardScaler(with_mean=True, with_std=True).fit_transform(ysamp.reshape(-1,1)).ravel()

    # Distance matrices
    DX = np.sqrt(((Xz[:,None,:] - Xz[None,:,:])**2).sum(axis=2))
    DY = np.abs(yz[:,None] - yz[None,:])
    # Normalize to [0,1]
    DX = DX / (DX.max() + 1e-12)
    DY = DY / (DY.max() + 1e-12)
    D  = alpha*DX + (1-alpha)*DY

    # Start with the farthest pair; break ties randomly
    i0, j0 = np.unravel_index(np.argmax(D), D.shape)
    if i0 == j0:
        i0 = rng.integers(0, N)
        j0 = (i0 + rng.integers(1, N)) % N

    selected = [i0, j0]
    remaining = set(range(N)) - set(selected)

    while len(selected) < Ncal:
        # For each remaining sample, compute its min distance to the selected set
        mins = []
        for r in remaining:
            mins.append((r, np.min(D[r, selected])))
        # pick the one with the largest "min distance"
        r_star = max(mins, key=lambda t: t[1])[0]
        selected.append(r_star)
        remaining.remove(r_star)

    cal_idx = np.array(sorted(selected), dtype=int)
    val_idx = np.array(sorted(list(remaining)), dtype=int)
    assert len(cal_idx) + len(val_idx) == N
    return cal_idx, val_idx


# ---------------------- MAIN ----------------------

def main():
    wn, avg_tr, all_tr, meta = load_data(
        data_dir=TRAIN_DATA_DIR,
        metadata_path=TRAIN_META_PATH,
        include_types=("DM1", "Control"),
        return_filenames=False,
        strict=False,
        report_samples=3,
    )
    # all_tr: (1731, 9N)  -> rows = spectra for scikit
    X_all = all_tr.T
    N_samples = meta.shape[0]

    # Build per-sample averaged spectra for SPXY distance calc (not for modeling)
    X_sample = avg_tr.T  # shape (N_samples, 1731)

    summary_rows = []

    for nice, tcol in TARGETS:
        print(f"\n=== SPXY validation for {nice} ({tcol}) ===")
        tgt_dir = os.path.join(OUT_DIR_SPXY, tcol)
        os.makedirs(tgt_dir, exist_ok=True)

        # mask samples with valid labels for this target
        keep = pd.to_numeric(meta[tcol], errors="coerce").notna().to_numpy()
        y_sample = meta.loc[keep, tcol].to_numpy(float)
        Xs = X_sample[keep]

        # Map kept samples to original indices
        orig_keep_idx = np.where(keep)[0]
        
        # groups for kept samples (needed for iPLS interval search on the CAL set)
        # (we’ll define groups_cal after we know cal_idx)
        # groups_kept = _groups_for_spectra(int(keep.sum()), reps=REPS)  # not needed here


        # winner row from leaderboard
        metrics_path = os.path.join(GLOBAL_RESULTS_DIR, tcol, f"{tcol}__metrics_all_combos.xlsx")
        dfm = pd.read_excel(metrics_path)
        # ---- add this guard RIGHT HERE ----
        if dfm is None or dfm.empty:
            raise FileNotFoundError(f"No rows in metrics file: {metrics_path}")
        winner = dfm.sort_values("rmse", ascending=True).iloc[0]
        model_name = str(winner["model"]).lower()
        methods = _parse_methods(str(winner["preprocessing"]))
        hp_str = str(winner["hp"])
        hp = _parse_hp(model_name, hp_str)

        for r in range(N_REPEATS):
            cal_idx, val_idx = _spxy_split(Xs, y_sample, cal_frac=CAL_FRAC, alpha=ALPHA, seed=42+r)

            # expand to spectra-level rows for modeling
            # (calibration and validation indices refer to SAMPLE rows)
            # after you get cal_idx, val_idx from _spxy_split(...)
            orig_cal_idx = orig_keep_idx[cal_idx]   # original sample indices
            orig_val_idx = orig_keep_idx[val_idx]
            
            # Build spectra row indices for CAL and VAL
            row_idx_cal = (orig_cal_idx[:, None] * REPS + np.arange(REPS)).ravel()
            row_idx_val = (orig_val_idx[:, None] * REPS + np.arange(REPS)).ravel()
            
            # Slice spectra
            Xcal = X_all[row_idx_cal]
            Xval = X_all[row_idx_val]
            
            # Labels


            ycal = np.repeat(y_sample[cal_idx].reshape(-1,1), REPS, axis=0)
            yval_true = y_sample[val_idx]  # sample-level truth

            groups_cal = _groups_for_spectra(len(cal_idx), reps=REPS)

            # train → predict spectra-level → collapse to sample
            yval_hat_spec = _train_predict(model_name, methods, hp, Xcal, ycal, Xval, groups_cal)
            yval_pred, yval_pred_sd = _sample_means_stds(yval_hat_spec, reps=REPS)

            mets = _metrics(yval_true, yval_pred)
            print(f"[repeat {r+1}] {tcol}: RMSE={mets['rmse']:.4f}, R²={mets['r2']:.4f}")

            # save artifacts
            title_suffix = f"{model_name.upper()} | " + (("+".join(methods)) if methods else "No Preprocessing") + f" | HP: {hp_str} | SPXY {int(CAL_FRAC*100)}/{int((1-CAL_FRAC)*100)}"
            parity_plot_sample_level(
                y_true_sample=yval_true.reshape(-1,1),
                y_pred_sample=yval_pred.reshape(-1,1),
                y_true_sample_std=None,
                y_pred_sample_std=yval_pred_sd.reshape(-1,1),
                target_names=(nice,),
                out_dir=tgt_dir,
                fname_prefix=f"spxy_parity_r{r+1}",
                dpi=600,
                title_suffix=title_suffix,
            )

            tidy_path = os.path.join(tgt_dir, f"{tcol}__spxy_predictions_r{r+1}.xlsx")
            # build a meta slice in the same order as val_idx
            meta_kept = meta.loc[keep].reset_index(drop=True)
            meta_val = meta_kept.iloc[val_idx].reset_index(drop=True)
            save_outer_predictions_excel(
                y_true_sample=yval_true.reshape(-1,1),
                y_pred_sample=yval_pred.reshape(-1,1),
                meta_kept=meta_val,
                target_order=(tcol,),
                out_path=tidy_path,
                y_true_sample_std=None,
                y_pred_sample_std=yval_pred_sd.reshape(-1,1),
            )

            with open(os.path.join(tgt_dir, f"{tcol}__spxy_metrics_r{r+1}.json"), "w") as f:
                json.dump(
                    {"model": model_name, "preprocessing": methods, "hp_str": hp_str, "hp": hp,
                     "alpha": ALPHA, "cal_frac": CAL_FRAC, "repeat": r+1, "metrics_sample_level": mets},
                    f, indent=2
                )

            summary_rows.append({
                "target": tcol, "repeat": r+1, "model": model_name, "preprocessing": "+".join(methods) if methods else "None",
                "hp": hp_str, "rmse": mets["rmse"], "r2": mets["r2"], "mae": mets["mae"], "medae": mets["medae"], "evs": mets["evs"]
            })

    # write summary table
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        df_sum.to_excel(os.path.join(OUT_DIR_SPXY, "spxy_summary.xlsx"), index=False)
        print(f"\n[OK] Wrote SPXY summary → {os.path.join(OUT_DIR_SPXY, 'spxy_summary.xlsx')}")


if __name__ == "__main__":
    main()
