# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:47:42 2025

@author: spect
"""

# -*- coding: utf-8 -*-
"""
Evaluate external (held-out) samples with the best per-target models
selected by your global 10-fold run.

For each target (SI, HGS, ADF):
  - Read the winner (model, preprocessing, hp) from the per-target metrics table
  - Retrain on FULL training set only (no CV), using the chosen preprocessing/hyperparams
  - Predict the EXTERNAL test set
  - Save: parity plot (with y-error bars from replicate predictions), Excel of predictions, metrics .json
"""

import os, re, json
import numpy as np
import pandas as pd
from pathlib import Path
# --- bootstrap project root onto sys.path ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # one level above /modules
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --- Your modules ---
from modules.data_loader import load_data                                   # :contentReference[oaicite:0]{index=0}
from modules.preprocessing import Preprocessing                              # :contentReference[oaicite:1]{index=1}
from modules.plot_parity import parity_plot_sample_level, save_outer_predictions_excel  # :contentReference[oaicite:2]{index=2}

# We will reuse internal helpers for iPLS interval selection
from modules.ten_fold_cv_regression_global import _best_interval_grouped     # :contentReference[oaicite:3]{index=3}

# scikit models
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score
import matplotlib.pyplot as plt

# --------------------------- CONFIG ---------------------------

# TRAIN set (same paths you used in main.py)
TRAIN_DATA_DIR  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\data"        # check
TRAIN_META_PATH = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata.csv"  # check

# EXTERNAL TEST set (you provided these)
TEST_META_PATH = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata_test_updated_in_order.csv"
TEST_DATA_DIR  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\data_test_updated"

# Where your global 10-fold run wrote per-target metrics ("...__metrics_all_combos.xlsx")
GLOBAL_RESULTS_DIR = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Matthews_code_Global\results_global_hp_per_target"

# Output root for this script
OUT_DIR_EXTERNAL = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\external_test_eval"
os.makedirs(OUT_DIR_EXTERNAL, exist_ok=True)

# Targets: (nice name, metadata column)
TARGETS = [
    ("Splicing Index", "target_SI"),
    ("Hand Grip Strength (%)", "HGS_pp_avg"),
    ("Average Ankle Dorsiflexion (%)", "ADF_pp_avg"),
]

# Number of replicate spectra per sample (your grid uses 9)
REPS = 9


# --------------------------- HELPERS ---------------------------

def _groups_for_spectra(n_samples: int, reps: int = 9) -> np.ndarray:
    return np.repeat(np.arange(n_samples, dtype=int), reps)

def _sample_means_stds(y_like: np.ndarray, reps: int = 9):
    """Return (means, stds) collapsing reps× spectra per sample."""
    y = np.asarray(y_like)
    assert (y.shape[0] % reps) == 0, "length must be multiple of REPS"
    N = y.shape[0] // reps
    if y.ndim == 1:
        y3 = y.reshape(N, reps, 1)
        m  = y3.mean(axis=1).ravel()
        s  = y3.std(axis=1)
        return m, s
    y3 = y.reshape(N, reps, -1)
    return y3.mean(axis=1), y3.std(axis=1)  # (N, K), (N, K)

def _fit_transform_pair(X_train_2D: np.ndarray, X_apply_2D: np.ndarray, methods: list[str]):
    """
    Apply preprocessing with TRAIN-only fit (EMSC ref, norm/SNV stats, etc.)
    Spectra are in rows for scikit; your Preprocessing expects (wn, nSpectra), so we transpose.
    """
    Xtr = X_train_2D.T.copy()  # (features, n_tr)
    Xap = X_apply_2D.T.copy()  # (features, n_ap)

    for m in (methods or []):
        if m == "EMSC":
            ref = np.mean(Xtr, axis=1)  # train-only reference  :contentReference[oaicite:4]{index=4}
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
        elif m in ("No Preprocessing", "No preprocessing", "no preprocessing"):
            pass
        elif m.strip() == "":
            pass
        else:
            raise ValueError(f"Unknown preprocessing method: {m}")

    return Xtr.T.astype(np.float32), Xap.T.astype(np.float32)

def _parse_methods(label: str) -> list[str]:
    if not label or label.lower().startswith("no"):
        return []
    return [p.strip() for p in label.split("+")]

def _parse_hp(model: str, hp_str: str):
    """
    Parse the string you stored in metrics for each model family.
     - PLS:  'C=10'
     - iPLS: 'C=9__I=10'
     - RF:   'ne150_md20_mss50_msl30_mfsqrt'
     - ACFNN:'hls256-128_actrelu_bs64_lr0.001_wd0.0001_mi200'
    Returns a dict with the canonical keys we need for scikit models,
    plus interval count for iPLS.
    """
    model = model.lower()

    if model == "pls":
        m = re.match(r"C=(\d+)", hp_str.strip())
        if not m: raise ValueError(f"Unrecognized PLS hp string: {hp_str}")
        return {"n_components": int(m.group(1))}

    if model == "ipls":
        m = re.match(r"C=(\d+)__I=(\d+)", hp_str.strip())
        if not m: raise ValueError(f"Unrecognized iPLS hp string: {hp_str}")
        return {"n_components": int(m.group(1)), "num_intervals": int(m.group(2))}

    if model == "rf":
        # ne{n_estimators}_md{max_depth}_mss{min_samples_split}_msl{min_samples_leaf}_mf{max_features}
        patt = r"ne(\d+)_md(\d+)_mss(\d+)_msl(\d+)_mf(\w+)"
        m = re.match(patt, hp_str.strip())
        if not m: raise ValueError(f"Unrecognized RF hp string: {hp_str}")
        n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = m.groups()
        return {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "min_samples_split": int(min_samples_split),
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": "sqrt" if max_features.lower()=="sqrt" else max_features,
        }

    if model == "acfnn":
        # hls256-128_actrelu_bs64_lr0.001_wd0.0001_mi200
        d = {}
        # hidden layers
        m = re.search(r"hls([0-9\-]+)", hp_str)
        if m:
            hls = tuple(int(x) for x in m.group(1).split("-"))
            d["hidden_layer_sizes"] = hls
        # activation
        m = re.search(r"act([a-zA-Z]+)", hp_str)
        d["activation"] = (m.group(1) if m else "relu")
        # batch size
        m = re.search(r"bs(\d+)", hp_str);      d["batch_size"] = int(m.group(1)) if m else 64
        # learning rate
        m = re.search(r"lr([0-9.]+)", hp_str);  d["learning_rate_init"] = float(m.group(1)) if m else 1e-3
        # weight decay (alpha)
        m = re.search(r"wd([0-9.]+)", hp_str);  d["alpha"] = float(m.group(1)) if m else 1e-4
        # max_iter
        m = re.search(r"mi(\d+)", hp_str);      d["max_iter"] = int(m.group(1)) if m else 300
        return d

    raise ValueError(f"Unknown model family: {model}")


def _train_predict_on_external(model_name: str,
                               methods: list[str],
                               hp: dict,
                               Xtr: np.ndarray, Ytr: np.ndarray,
                               Xte: np.ndarray,
                               groups_train: np.ndarray):
    """
    Train on full TRAIN set (9N rows) and predict EXTERNAL (9M rows).
    Returns y_pred spectra-level (len=9M) as 1D array.
    """
    model_name = model_name.lower()

    if model_name == "pls":
        # Preprocess
        Xt_tr, Xt_te = _fit_transform_pair(Xtr, Xte, methods)
        # Fit single-output PLS
        pls = PLSRegression(n_components=int(hp["n_components"]))
        pls.fit(Xt_tr, Ytr)
        yhat = pls.predict(Xt_te).ravel()
        return yhat

    if model_name == "rf":
        # Preprocess
        Xt_tr, Xt_te = _fit_transform_pair(Xtr, Xte, methods)
        rf = RandomForestRegressor(
            n_estimators=int(hp["n_estimators"]),
            max_depth=int(hp["max_depth"]),
            min_samples_split=int(hp["min_samples_split"]),
            min_samples_leaf=int(hp["min_samples_leaf"]),
            max_features=hp["max_features"],
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(Xt_tr, Ytr.ravel())
        yhat = rf.predict(Xt_te)
        return yhat

    if model_name == "acfnn":
        # Preprocess
        Xt_tr, Xt_te = _fit_transform_pair(Xtr, Xte, methods)
        mlp = MLPRegressor(random_state=42, **hp)
        mlp.fit(Xt_tr, Ytr.ravel())
        yhat = mlp.predict(Xt_te)
        return yhat

    if model_name == "ipls":
        # Select BEST interval on full TRAIN set with grouping, then fit PLS there
        num_intervals = int(hp["num_intervals"])
        n_components  = int(hp["n_components"])

        # Find winning interval indices (a,b) on TRAIN only
        (a,b), _ = _best_interval_grouped(
            X_tr=Xtr, Y_tr=Ytr, groups_tr=groups_train,
            preprocess_methods=methods,
            n_components=n_components, num_intervals=num_intervals,
            reps=REPS, n_splits=max(3, min(5, int(len(np.unique(groups_train))))),
            random_state=42
        )  # :contentReference[oaicite:5]{index=5}

        # Apply preprocessing inside interval only
        Xt_tr, Xt_te = _fit_transform_pair(Xtr[:, a:b], Xte[:, a:b], methods)
        pls = PLSRegression(n_components=n_components)
        pls.fit(Xt_tr, Ytr)
        yhat = pls.predict(Xt_te).ravel()
        return yhat

    raise ValueError(f"Unsupported model for external eval: {model_name}")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "evs": float(explained_variance_score(y_true, y_pred)),
    }


# --------------------------- MAIN ---------------------------

def main():
    # 1) Load TRAIN + TEST with consistent alignment & safety checks
    #    (same loader you already use)  :contentReference[oaicite:6]{index=6}
    #    We drop rows with missing target per-target when we build Y below.
    wn_tr, avg_tr, all_tr, meta_tr = load_data(
        data_dir=TRAIN_DATA_DIR,
        metadata_path=TRAIN_META_PATH,
        include_types=("DM1", "Control"),
        return_filenames=False,
        strict=False,
        report_samples=3,
    )
    wn_te, avg_te, all_te, meta_te = load_data(
        data_dir=TEST_DATA_DIR,
        metadata_path=TEST_META_PATH,
        include_types=("DM1", "Control"),
        return_filenames=False,
        strict=False,
        report_samples=3,
    )
    # Shapes: all_* are (1731, 9N). Transpose to (9N, 1731) for model fit.
    Xtr_full = all_tr.T
    Xte_full = all_te.T

    # Replicate grouping for TRAIN (needed by iPLS interval selection)
    Ntr = meta_tr.shape[0]
    groups_train = _groups_for_spectra(Ntr, reps=REPS)

    # 2) For each target, read the winner row from your metrics file written by main.py  :contentReference[oaicite:7]{index=7}
    for nice_name, tcol in TARGETS:
        print(f"\n=== External evaluation for {nice_name} ({tcol}) ===")

        # Per-target metrics (already written by your global run)  :contentReference[oaicite:8]{index=8}
        metrics_path = os.path.join(GLOBAL_RESULTS_DIR, tcol, f"{tcol}__metrics_all_combos.xlsx")
        dfm = pd.read_excel(metrics_path)
        if dfm.empty:
            print(f"[WARN] No metrics found for {tcol}: {metrics_path}")
            continue
        winner = dfm.sort_values("rmse", ascending=True).iloc[0]
        model_name   = str(winner["model"]).lower()
        methods_lab  = str(winner["preprocessing"])
        hp_str       = str(winner["hp"])
        methods_list = _parse_methods(methods_lab)
        hp           = _parse_hp(model_name, hp_str)

        # Build y arrays (TRAIN and TEST) for THIS target
        # Use meta order from loader; drop NaNs for the target.
        tr_keep = pd.to_numeric(meta_tr[tcol], errors="coerce").notna().to_numpy()
        te_keep = pd.to_numeric(meta_te[tcol], errors="coerce").notna().to_numpy()

        Xtr = Xtr_full[np.repeat(tr_keep, REPS)]
        Xte = Xte_full[np.repeat(te_keep, REPS)]
        ytr = meta_tr.loc[tr_keep, tcol].to_numpy(float)
        yte = meta_te.loc[te_keep, tcol].to_numpy(float)

        # Repeat ytr to spectra level (9 per sample, col vector)
        Ytr = np.repeat(ytr.reshape(-1, 1), REPS, axis=0)

        # Update groups to reflect any dropped TRAIN rows
        groups_tr_kept = _groups_for_spectra(int(tr_keep.sum()), reps=REPS)

        # Fit the chosen model on FULL TRAIN and predict TEST (spectra level)
        yhat_te_spectra = _train_predict_on_external(
            model_name=model_name,
            methods=methods_list,
            hp=hp,
            Xtr=Xtr, Ytr=Ytr,
            Xte=Xte,
            groups_train=groups_tr_kept,
        )

        # Collapse to sample level for TEST
        y_pred_s, y_pred_std = _sample_means_stds(yhat_te_spectra, reps=REPS)
        y_true_s = yte  # single label per sample in metadata

        # Metrics (sample level)
        mets = _compute_metrics(y_true_s, y_pred_s)

        # 3) Save artifacts
        tgt_dir = os.path.join(OUT_DIR_EXTERNAL, tcol)
        os.makedirs(tgt_dir, exist_ok=True)

        # 3a) Parity plot (y-error bars from replicate preds)
        title_suffix = f"{model_name.upper()} | " + (methods_lab if methods_list else "No Preprocessing") + f" | HP: {hp_str}"
        parity_plot_sample_level(
            y_true_sample=y_true_s.reshape(-1,1),
            y_pred_sample=y_pred_s.reshape(-1,1),
            y_true_sample_std=None,                # ground-truth has no replicate std
            y_pred_sample_std=y_pred_std.reshape(-1,1),
            target_names=(nice_name,),
            out_dir=tgt_dir,
            fname_prefix="external_parity",
            dpi=600,
            title_suffix=title_suffix,
            
        )  # :contentReference[oaicite:9]{index=9}
        plt.show()
        # 3b) Tidy Excel (one sheet)
        tidy_path = os.path.join(tgt_dir, f"{tcol}__external_predictions.xlsx")
        save_outer_predictions_excel(
            y_true_sample=y_true_s.reshape(-1,1),
            y_pred_sample=y_pred_s.reshape(-1,1),
            meta_kept=meta_te.loc[te_keep].reset_index(drop=True),
            target_order=(tcol,),
            out_path=tidy_path,
            y_true_sample_std=None,
            y_pred_sample_std=y_pred_std.reshape(-1,1),
        )  # :contentReference[oaicite:10]{index=10}

        # 3c) Metrics JSON
        with open(os.path.join(tgt_dir, f"{tcol}__external_metrics.json"), "w") as f:
            json.dump(
                {
                    "model": model_name,
                    "preprocessing": methods_list,
                    "hp_str": hp_str,
                    "hp": hp,
                    "metrics_sample_level": mets,
                },
                f, indent=2
            )

        print(f"[OK] {tcol}: RMSE={mets['rmse']:.4f}, R²={mets['r2']:.4f}")
        print(f"     plots → {os.path.join(tgt_dir, 'external_parity_1.png')}")
        print(f"     excel → {tidy_path}")

if __name__ == "__main__":
    main()
