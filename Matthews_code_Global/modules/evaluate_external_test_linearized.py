# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:27:18 2025

@author: spect
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:47:42 2025

@author: spect

Evaluate external (held-out) samples with the best per-target models
selected by your global 10-fold run, then apply a tiny post-hoc linear
calibration transfer:

    y_true ≈ a + b * y_pred

Calibration modes:
  - CAL_MODE = "loo": per-sample leave-one-out calibration on external set
  - CAL_MODE = "calibrants": fit a,b on explicit external calibrants by Sample_ID,
                              apply to the rest; metrics reported on non-calibrants.
"""

import os, re, json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score
import matplotlib.pyplot as plt

# --- bootstrap project root onto sys.path ---
ROOT = Path(__file__).resolve().parents[1]  # one level above /modules
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Your modules ---
from modules.data_loader import load_data
from modules.preprocessing import Preprocessing
from modules.plot_parity import parity_plot_sample_level, save_outer_predictions_excel
from modules.ten_fold_cv_regression_global import _best_interval_grouped

# --------------------------- CONFIG ---------------------------

# TRAIN set (same paths you used in main.py)
TRAIN_DATA_DIR  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\data"
TRAIN_META_PATH = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata.csv"

# EXTERNAL TEST set
TEST_META_PATH  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata_test_updated_in_order.csv"
TEST_DATA_DIR   = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\data_test_updated"

# Where your global 10-fold run wrote per-target metrics ("...__metrics_all_combos.xlsx")
GLOBAL_RESULTS_DIR = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Matthews_code_Global\results_global_hp_per_target"

# Output root for this script
OUT_DIR_EXTERNAL = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\external_test_eval_linearized"
os.makedirs(OUT_DIR_EXTERNAL, exist_ok=True)

# Targets: (nice name, metadata column)
TARGETS = [
    ("Splicing Index", "target_SI"),
    ("Hand Grip Strength (%)", "HGS_pp_avg"),
    ("Average Ankle Dorsiflexion (%)", "ADF_pp_avg"),
]

# Number of replicate spectra per sample
REPS = 9

# ---- Calibration settings ----
# "loo" (default) or "calibrants"
CAL_MODE = "loo"
# Only used when CAL_MODE == "calibrants". Put Sample_ID strings from meta_te here if you want.
CALIBRANT_IDS = []  # e.g., ["DM1_006", "DM1_009", "AdCo_002"]
MIN_CAL_SAMPLES = 3      # require at least this many points to fit a,b
EPS_VAR = 1e-12          # guard against zero-variance yhat in linear fit


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
    """
    Apply preprocessing with TRAIN-only fit (EMSC ref, norm/SNV stats, etc.)
    Spectra are in rows for scikit; your Preprocessing expects (wn, nSpectra), so we transpose.
    """
    Xtr = X_train_2D.T.copy()  # (features, n_tr)
    Xap = X_apply_2D.T.copy()  # (features, n_ap)

    for m in (methods or []):
        if m == "EMSC":
            ref = np.mean(Xtr, axis=1)  # train-only reference
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

def _train_predict_on_external(model_name: str,
                               methods: list[str],
                               hp: dict,
                               Xtr: np.ndarray, Ytr: np.ndarray,
                               Xte: np.ndarray,
                               groups_train: np.ndarray) -> np.ndarray:
    model_name = model_name.lower()

    if model_name == "pls":
        Xt_tr, Xt_te = _fit_transform_pair(Xtr, Xte, methods)
        pls = PLSRegression(n_components=int(hp["n_components"]))
        pls.fit(Xt_tr, Ytr)
        return pls.predict(Xt_te).ravel()

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
        return rf.predict(Xt_te)

    if model_name == "acfnn":
        Xt_tr, Xt_te = _fit_transform_pair(Xtr, Xte, methods)
        mlp = MLPRegressor(random_state=42, **hp)
        mlp.fit(Xt_tr, Ytr.ravel())
        return mlp.predict(Xt_te)

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
        return pls.predict(Xt_te).ravel()

    raise ValueError(f"Unsupported model for external eval: {model_name}")

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "evs":  float(explained_variance_score(y_true, y_pred)),
    }

def _fit_line(yhat: np.ndarray, ytrue: np.ndarray):
    """Fit ytrue ≈ a + b * yhat using ordinary least squares. Returns (a,b)."""
    yhat = np.asarray(yhat).ravel()
    ytrue = np.asarray(ytrue).ravel()
    if len(yhat) < MIN_CAL_SAMPLES or np.var(yhat) < EPS_VAR:
        # Not enough variation to fit; fallback to identity
        return 0.0, 1.0
    # Solve [1, yhat] * [a, b] = ytrue
    X = np.column_stack([np.ones_like(yhat), yhat])
    # OLS via lstsq (stable for tiny n)
    beta, *_ = np.linalg.lstsq(X, ytrue, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    return a, b

def _apply_line(yhat: np.ndarray, a: float, b: float):
    return a + b * np.asarray(yhat)

def _loo_calibrate(yhat_s: np.ndarray, ytrue_s: np.ndarray):
    """Return calibrated preds, and per-sample (a,b) fitted on the other samples."""
    n = len(yhat_s)
    ycal = np.zeros_like(yhat_s, dtype=float)
    a_list, b_list = np.zeros(n), np.ones(n)
    idx = np.arange(n)
    for j in range(n):
        mask = idx != j
        a, b = _fit_line(yhat_s[mask], ytrue_s[mask])
        a_list[j], b_list[j] = a, b
        ycal[j] = _apply_line(yhat_s[j], a, b)
    return ycal, a_list, b_list

def _calibrants_calibrate(yhat_s: np.ndarray, ytrue_s: np.ndarray, sample_ids: np.ndarray, calibrant_ids: list[str]):
    """Fit a,b on calibrants; apply to all. Return ycal, global_a, global_b, mask_eval (non-calibrants)."""
    cal_mask = np.isin(sample_ids, np.array(calibrant_ids, dtype=object))
    a, b = _fit_line(yhat_s[cal_mask], ytrue_s[cal_mask])
    ycal = _apply_line(yhat_s, a, b)
    eval_mask = ~cal_mask  # report metrics on non-calibrants
    return ycal, a, b, cal_mask, eval_mask

# --------------------------- MAIN ---------------------------

def main():
    # 1) Load TRAIN + TEST with consistent alignment & safety checks
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

    for nice_name, tcol in TARGETS:
        print(f"\n=== External evaluation for {nice_name} ({tcol}) ===")

        # Winner row from metrics
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

        # Keep rows with target present
        tr_keep = pd.to_numeric(meta_tr[tcol], errors="coerce").notna().to_numpy()
        te_keep = pd.to_numeric(meta_te[tcol], errors="coerce").notna().to_numpy()

        Xtr = Xtr_full[np.repeat(tr_keep, REPS)]
        Xte = Xte_full[np.repeat(te_keep, REPS)]
        ytr = meta_tr.loc[tr_keep, tcol].to_numpy(float)
        yte = meta_te.loc[te_keep, tcol].to_numpy(float)

        # Repeat ytr to spectra level (9 per sample, col vector)
        Ytr = np.repeat(ytr.reshape(-1, 1), REPS, axis=0)
        groups_tr_kept = _groups_for_spectra(int(tr_keep.sum()), reps=REPS)

        # Fit chosen model on FULL TRAIN and predict TEST (spectra level)
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
        y_true_s = yte  # per-sample label

        # -------- Raw metrics & plot --------
        mets_raw = _compute_metrics(y_true_s, y_pred_s)

        tgt_dir = os.path.join(OUT_DIR_EXTERNAL, tcol)
        os.makedirs(tgt_dir, exist_ok=True)

        title_suffix = f"{model_name.upper()} | " + (methods_lab if methods_list else "No Preprocessing") + f" | HP: {hp_str}"

        parity_plot_sample_level(
            y_true_sample=y_true_s.reshape(-1,1),
            y_pred_sample=y_pred_s.reshape(-1,1),
            y_true_sample_std=None,
            y_pred_sample_std=y_pred_std.reshape(-1,1),
            target_names=(nice_name,),
            out_dir=tgt_dir,
            fname_prefix="external_parity_raw",
            dpi=600,
            title_suffix=title_suffix + " | RAW",
        )
        plt.show()

        # -------- Linear calibration --------
        sample_ids = meta_te.loc[te_keep, "Sample_ID"].astype(str).to_numpy()
        cal_artifacts = {}

        if CAL_MODE.lower() == "calibrants" and len(CALIBRANT_IDS) >= MIN_CAL_SAMPLES:
            ycal_s, a, b, cal_mask, eval_mask = _calibrants_calibrate(
                y_pred_s, y_true_s, sample_ids, CALIBRANT_IDS
            )
            mets_cal = _compute_metrics(y_true_s[eval_mask], ycal_s[eval_mask])
            cal_artifacts.update({
                "mode": "calibrants",
                "global_a": float(a), "global_b": float(b),
                "n_calibrants": int(cal_mask.sum()),
                "calibrant_ids": list(np.array(sample_ids)[cal_mask]),
            })
        else:
            # Default (and safe) LOO calibration
            ycal_s, a_vec, b_vec = _loo_calibrate(y_pred_s, y_true_s)
            mets_cal = _compute_metrics(y_true_s, ycal_s)
            cal_artifacts.update({
                "mode": "loo",
                "per_sample_a": [float(x) for x in a_vec],
                "per_sample_b": [float(x) for x in b_vec],
            })

        # -------- Calibrated plot --------
        parity_plot_sample_level(
            y_true_sample=y_true_s.reshape(-1,1),
            y_pred_sample=ycal_s.reshape(-1,1),
            y_true_sample_std=None,
            y_pred_sample_std=y_pred_std.reshape(-1,1),
            target_names=(nice_name,),
            out_dir=tgt_dir,
            fname_prefix="external_parity_calibrated",
            dpi=600,
            title_suffix=title_suffix + " | CALIBRATED",
        )
        plt.show()

        # -------- Excel (raw + calibrated) --------
        tidy_path = os.path.join(tgt_dir, f"{tcol}__external_predictions.xlsx")
        meta_kept = meta_te.loc[te_keep].reset_index(drop=True).copy()
        
        # First write the standard sheet using your helper
        save_outer_predictions_excel(
            y_true_sample=y_true_s.reshape(-1,1),
            y_pred_sample=y_pred_s.reshape(-1,1),
            meta_kept=meta_kept,
            target_order=(tcol,),
            out_path=tidy_path,
            y_true_sample_std=None,
            y_pred_sample_std=y_pred_std.reshape(-1,1),
        )
        
        # Now append a new sheet with calibrated outputs using openpyxl
        df_extra = pd.DataFrame({
            "Sample_ID": meta_kept["Sample_ID"].astype(str).values,
            f"{tcol}__y_true": y_true_s,
            f"{tcol}__y_pred_raw": y_pred_s,
            f"{tcol}__y_pred_calibrated": ycal_s,
            f"{tcol}__y_pred_std": y_pred_std,
        })
        
        if cal_artifacts.get("mode") == "loo":
            df_extra["cal_a"] = cal_artifacts["per_sample_a"]
            df_extra["cal_b"] = cal_artifacts["per_sample_b"]
        elif cal_artifacts.get("mode") == "calibrants":
            df_extra["cal_a"] = cal_artifacts["global_a"]
            df_extra["cal_b"] = cal_artifacts["global_b"]
            df_extra["is_calibrant"] = np.isin(
                df_extra["Sample_ID"].values, np.array(CALIBRANT_IDS, dtype=object)
            )
        
        # Append with openpyxl; fallback if not installed
        try:
            with pd.ExcelWriter(tidy_path, mode="a", engine="openpyxl",
                                if_sheet_exists="replace") as xlw:
                df_extra.to_excel(xlw, sheet_name="calibrated_preds", index=False)
        except Exception as e:
            # Fallback: write a separate calibrated file to avoid xlsxwriter append issues
            alt_path = os.path.join(tgt_dir, f"{tcol}__external_predictions_CALIBRATED.xlsx")
            with pd.ExcelWriter(alt_path, engine="xlsxwriter") as xlw:
                df_extra.to_excel(xlw, sheet_name="calibrated_preds", index=False)
            print(f"[WARN] Could not append to {tidy_path} ({e}). "
                  f"Wrote calibrated sheet to {alt_path} instead.")


        # Also write calibration params as CSV for quick diffing
        if cal_artifacts.get("mode") == "loo":
            pd.DataFrame({
                "Sample_ID": meta_kept["Sample_ID"].astype(str).values,
                "a": cal_artifacts["per_sample_a"],
                "b": cal_artifacts["per_sample_b"],
            }).to_csv(os.path.join(tgt_dir, f"{tcol}__external_calibration_params.csv"), index=False)
        else:
            pd.DataFrame({
                "mode": ["calibrants"],
                "a": [cal_artifacts["global_a"]],
                "b": [cal_artifacts["global_b"]],
                "n_calibrants": [len(CALIBRANT_IDS)],
                "calibrant_ids": [", ".join(CALIBRANT_IDS)],
            }).to_csv(os.path.join(tgt_dir, f"{tcol}__external_calibration_params.csv"), index=False)

        # -------- Metrics JSON --------
        with open(os.path.join(tgt_dir, f"{tcol}__external_metrics.json"), "w") as f:
            json.dump(
                {
                    "model": model_name,
                    "preprocessing": methods_list,
                    "hp_str": hp_str,
                    "hp": hp,
                    "metrics_sample_level_raw": mets_raw,
                    "metrics_sample_level_calibrated": mets_cal,
                    "calibration": cal_artifacts,
                },
                f, indent=2
            )

        print(f"[OK] {tcol} RAW: RMSE={mets_raw['rmse']:.4f}, R²={mets_raw['r2']:.4f}")
        print(f"     {tcol} CAL: RMSE={mets_cal['rmse']:.4f}, R²={mets_cal['r2']:.4f}")
        print(f"     plots → {os.path.join(tgt_dir, 'external_parity_raw_1.png')}")
        print(f"              {os.path.join(tgt_dir, 'external_parity_calibrated_1.png')}")
        print(f"     excel → {tidy_path}")

if __name__ == "__main__":
    main()
