# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 10:13:47 2025

@author: spect
"""

# -*- coding: utf-8 -*-
"""
Evaluate EXTERNAL (held-out) set for EVERY hyperparameter combination seen in your
global 10-fold run, then:
  - Save a CSV of metrics for ALL combos on the external set
  - Save parity plots for the TOP-K (by RMSE) per target

This script mirrors your pipeline:
  * preprocessing fit on TRAIN only
  * iPLS: interval re-selected on TRAIN only (no leakage)
  * replicate-aware collapse to sample level for metrics/plots
"""

import os, re, json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- BOOTSTRAP ---------------------------
ROOT = Path(__file__).resolve().parents[1]  # one level above /modules
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Your modules
from modules.data_loader import load_data
from modules.preprocessing import Preprocessing
from modules.plot_parity import parity_plot_sample_level
from modules.ten_fold_cv_regression_global import _best_interval_grouped

# --- scikit
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

# --------------------------- CONFIG ---------------------------

# TRAIN set
TRAIN_DATA_DIR  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\data"
TRAIN_META_PATH = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata.csv"

# EXTERNAL TEST set
TEST_DATA_DIR   = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\data_test_updated"
TEST_META_PATH  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata_test_updated_in_order.csv"

# Global 10-fold results (where the __metrics_all_combos.xlsx lives)
GLOBAL_RESULTS_DIR = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Matthews_code_Global\results_global_hp_per_target"

# # Output root OG
# OUT_DIR_EXTERNAL = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\external_test_eval_ALL"
# os.makedirs(OUT_DIR_EXTERNAL, exist_ok=True)
# Output root (parity plots + per-target CSVs + top-K JSONs)
OUT_DIR_EXTERNAL = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\external_test_eval\With controls - rerun"
os.makedirs(OUT_DIR_EXTERNAL, exist_ok=True)
# Targets: (nice name, metadata column)
TARGETS = [
    ("Splicing Index", "target_SI"),
    ("Hand Grip Strength (%)", "HGS_pp_avg"),
    ("Average Ankle Dorsiflexion (%)", "ADF_pp_avg"),
]

# Replicates per sample
REPS = 9

# Save top-K parity plots by RMSE
TOPK = 1

# --------------------------- HELPERS ---------------------------

def _groups_for_spectra(n_samples: int, reps: int = 9) -> np.ndarray:
    return np.repeat(np.arange(n_samples, dtype=int), reps)

def _sample_means_stds(y_like: np.ndarray, reps: int = 9):
    y = np.asarray(y_like)
    assert (y.shape[0] % reps) == 0, "length must be multiple of REPS"
    N = y.shape[0] // reps
    if y.ndim == 1:
        y3 = y.reshape(N, reps, 1)
        m  = y3.mean(axis=1).ravel()
        s  = y3.std(axis=1).ravel()
        return m, s
    y3 = y.reshape(N, reps, -1)
    return y3.mean(axis=1), y3.std(axis=1)

def _fit_transform_pair(X_train_2D: np.ndarray, X_apply_2D: np.ndarray, methods: list[str]):
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
        d = {}
        m = re.search(r"hls([0-9\-]+)", hp_str)
        if m:
            hls = tuple(int(x) for x in m.group(1).split("-"))
            d["hidden_layer_sizes"] = hls
        m = re.search(r"act([a-zA-Z]+)", hp_str)
        d["activation"] = (m.group(1) if m else "relu")
        m = re.search(r"bs(\d+)", hp_str);      d["batch_size"] = int(m.group(1)) if m else 64
        m = re.search(r"lr([0-9.]+)", hp_str);  d["learning_rate_init"] = float(m.group(1)) if m else 1e-3
        m = re.search(r"wd([0-9.]+)", hp_str);  d["alpha"] = float(m.group(1)) if m else 1e-4
        m = re.search(r"mi(\d+)", hp_str);      d["max_iter"] = int(m.group(1)) if m else 300
        return d

    raise ValueError(f"Unknown model family: {model}")

def _train_predict(model_name: str, methods: list[str], hp: dict,
                   Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray,
                   groups_train: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return spectra-level predictions for TEST and extras (like chosen interval)."""
    model_name = model_name.lower()
    extras = {}

    if model_name == "pls":
        Xt_tr, Xt_te = _fit_transform_pair(Xtr, Xte, methods)
        pls = PLSRegression(n_components=int(hp["n_components"]))
        pls.fit(Xt_tr, Ytr)
        return pls.predict(Xt_te).ravel(), extras

    if model_name == "rf":
        Xt_tr, Xt_te = _fit_transform_pair(Xtr, Xte, methods)
        rf = RandomForestRegressor(
            n_estimators=int(hp["n_estimators"]),
            max_depth=int(hp["max_depth"]),
            min_samples_split=int(hp["min_samples_split"]),
            min_samples_leaf=int(hp["min_samples_leaf"]),
            max_features=hp["max_features"],
            random_state=42, n_jobs=-1,
        )
        rf.fit(Xt_tr, Ytr.ravel())
        return rf.predict(Xt_te), extras

    if model_name == "acfnn":
        Xt_tr, Xt_te = _fit_transform_pair(Xtr, Xte, methods)
        mlp = MLPRegressor(random_state=42, **hp)
        mlp.fit(Xt_tr, Ytr.ravel())
        return mlp.predict(Xt_te), extras

    if model_name == "ipls":
        num_intervals = int(hp["num_intervals"])
        n_components  = int(hp["n_components"])
        (a,b), _ = _best_interval_grouped(
            X_tr=Xtr, Y_tr=Ytr, groups_tr=groups_train,
            preprocess_methods=methods,
            n_components=n_components, num_intervals=num_intervals,
            reps=REPS, n_splits=max(3, min(5, int(len(np.unique(groups_train))))),
            random_state=42
        )
        Xt_tr, Xt_te = _fit_transform_pair(Xtr[:, a:b], Xte[:, a:b], methods)
        pls = PLSRegression(n_components=n_components)
        pls.fit(Xt_tr, Ytr)
        extras.update({"interval_a": int(a), "interval_b": int(b)})
        return pls.predict(Xt_te).ravel(), extras

    raise ValueError(f"Unsupported model: {model_name}")

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    # Line fit for plot annotation/debug
    if (np.std(y_true) > 0) and (np.std(y_pred) > 0):
        slope, intercept = np.polyfit(y_true, y_pred, 1)
    else:
        slope, intercept = np.nan, np.nan
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "evs": float(explained_variance_score(y_true, y_pred)),
        "slope": float(slope),
        "intercept": float(intercept),
    }

def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

# --------------------------- MAIN ---------------------------

def main():
    # Load TRAIN + TEST once
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

    # Shapes: all_* are (features, 9N). We want (9N, features)
    Xtr_full = all_tr.T
    Xte_full = all_te.T

    Ntr = meta_tr.shape[0]
    groups_train_full = _groups_for_spectra(Ntr, reps=REPS)

    for nice_name, tcol in TARGETS:
        print(f"\n=== External ALL-COMBO evaluation for {nice_name} ({tcol}) ===")

        # Read ALL combos from your global run
        metrics_path = os.path.join(GLOBAL_RESULTS_DIR, tcol, f"{tcol}__metrics_all_combos.xlsx")
        dfm = pd.read_excel(metrics_path)
        if dfm.empty:
            print(f"[WARN] No combos found for {tcol}: {metrics_path}")
            continue

        combos = dfm[["model", "preprocessing", "hp"]].drop_duplicates().reset_index(drop=True)
        print(f"[INFO] {tcol}: evaluating {len(combos)} unique combos on EXTERNAL")

        # Build per-target datasets (drop NaNs in target)
        tr_keep = pd.to_numeric(meta_tr[tcol], errors="coerce").notna().to_numpy()
        te_keep = pd.to_numeric(meta_te[tcol], errors="coerce").notna().to_numpy()

        Xtr = Xtr_full[np.repeat(tr_keep, REPS)]
        Xte = Xte_full[np.repeat(te_keep, REPS)]
        ytr = meta_tr.loc[tr_keep, tcol].to_numpy(float)
        yte = meta_te.loc[te_keep, tcol].to_numpy(float)

        Ytr = np.repeat(ytr.reshape(-1, 1), REPS, axis=0)
        groups_tr_kept = _groups_for_spectra(int(tr_keep.sum()), reps=REPS)

        tgt_dir = os.path.join(OUT_DIR_EXTERNAL, tcol)
        os.makedirs(tgt_dir, exist_ok=True)

        # Evaluate every combo, collect metrics & (optionally) predictions for top-K selection
        records = []
        preds_cache = []  # store (idx, y_pred_s, y_pred_std, title_suffix)

        for idx, row in combos.iterrows():
            model_name = str(row["model"]).lower()
            methods_lab = str(row["preprocessing"])
            hp_str      = str(row["hp"])
            methods     = _parse_methods(methods_lab)
            hp          = _parse_hp(model_name, hp_str)

            try:
                yhat_te_spectra, extras = _train_predict(
                    model_name=model_name,
                    methods=methods,
                    hp=hp,
                    Xtr=Xtr, Ytr=Ytr,
                    Xte=Xte,
                    groups_train=groups_tr_kept,
                )
            except Exception as e:
                print(f"[ERR] {tcol} | {model_name} | {methods_lab} | {hp_str}: {e}")
                rec = {
                    "model": model_name, "preprocessing": methods_lab, "hp": hp_str,
                    "rmse": np.nan, "r2": np.nan, "mae": np.nan, "medae": np.nan, "evs": np.nan,
                    "slope": np.nan, "intercept": np.nan, **extras
                }
                records.append(rec)
                continue

            y_pred_s, y_pred_std = _sample_means_stds(yhat_te_spectra, reps=REPS)
            y_true_s = yte
            mets = _compute_metrics(y_true_s, y_pred_s)

            rec = {
                "model": model_name, "preprocessing": methods_lab, "hp": hp_str,
                **mets, **extras
            }
            records.append(rec)

            # keep preds in cache for potential plotting later
            title_suffix = f"{model_name.upper()} | " + (methods_lab if methods else "No Preprocessing") + f" | HP: {hp_str}"
            preds_cache.append((idx, y_true_s.copy(), y_pred_s.copy(), y_pred_std.copy(), title_suffix))

        # Save ALL metrics to CSV
        df_all = pd.DataFrame(records)
        csv_path = os.path.join(tgt_dir, f"{tcol}__external_metrics_ALL.csv")
        df_all.to_csv(csv_path, index=False)

        # Pick TOP-K by RMSE (ascending), ignoring NaNs
        df_sorted = df_all.sort_values(["rmse", "mae", "r2"], ascending=[True, True, False])
        df_topk = df_sorted.dropna(subset=["rmse"]).head(TOPK).reset_index(drop=True)

        print(f"[INFO] {tcol}: top-{TOPK} by RMSE")
        print(df_topk[["model","preprocessing","hp","rmse","r2"]])

        # Make parity plots for TOP-K
        for rank in range(min(TOPK, len(df_topk))):
            row = df_topk.iloc[rank]
            # find cached preds by combo index
            # We match by unique triple (model, preprocessing, hp)
            def _matches(p):
                _, _, _, _, title = p
                return (row["model"].upper() in title) and (str(row["preprocessing"]) in title) and (str(row["hp"]) in title)

            # If multiple matches (unlikely), just take the first
            match = next((p for p in preds_cache if _matches(p)), None)
            if match is None:
                # Fallback: recompute predictions for this combo (should be rare)
                model_name = row["model"]; methods_lab = row["preprocessing"]; hp_str = row["hp"]
                methods = _parse_methods(methods_lab); hp = _parse_hp(model_name, hp_str)
                yhat_te_spectra, _ = _train_predict(model_name, methods, hp, Xtr, Ytr, Xte, groups_tr_kept)
                y_pred_s, y_pred_std = _sample_means_stds(yhat_te_spectra, reps=REPS)
                y_true_s = yte
                title_suffix = f"{model_name.upper()} | " + (methods_lab if methods else "No Preprocessing") + f" | HP: {hp_str}"
            else:
                _, y_true_s, y_pred_s, y_pred_std, title_suffix = match

            fname = f"external_parity_TOP{rank+1}_{_sanitize(row['model'])}__{_sanitize(str(row['preprocessing']))}__{_sanitize(str(row['hp']))}"
            parity_plot_sample_level(
                y_true_sample=y_true_s.reshape(-1,1),
                y_pred_sample=y_pred_s.reshape(-1,1),
                y_true_sample_std=None,
                y_pred_sample_std=y_pred_std.reshape(-1,1),
                target_names=(nice_name,),
                out_dir=tgt_dir,
                fname_prefix=fname,
                dpi=600,
                title_suffix=title_suffix + " | TOP",
            )
            plt.show()

        # Also dump a small JSON index with the top-K rows for convenience
        with open(os.path.join(tgt_dir, f"{tcol}__external_TOP{TOPK}.json"), "w") as f:
            json.dump(df_topk.to_dict(orient="records"), f, indent=2)

        print(f"[OK] {tcol}: wrote {csv_path} and {len(df_topk)} parity plots")

if __name__ == "__main__":
    main()
