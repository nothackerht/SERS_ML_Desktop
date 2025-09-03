# -*- coding: utf-8 -*-
"""
main.py — GLOBAL-HP 10-fold CV (PLS / RF / iPLS / AC-FNN), per-target selection
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as _mse, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

from modules.ten_fold_cv_regression_global import GlobalGroupedCV
from modules.data_loader import load_data
from modules.plot_parity import parity_plot_sample_level, save_outer_predictions_excel

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
META_PATH = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"
OUT_DIR   = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Matthews_code_Global\results_global_hp"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
INCLUDE_TYPES = ("DM1", "Control")

wavenumbers, averaged_spectra, all_spectra, filenames_per_column, meta = load_data(
    data_dir=DATA_DIR,
    metadata_path=META_PATH,
    include_types=INCLUDE_TYPES,
    return_filenames=True,
    strict=False,
    report_samples=5,
)
print(f"[SHAPES] wav: {wavenumbers.shape} | averaged: {averaged_spectra.shape} | all: {all_spectra.shape}")
print(f"[META] rows={len(meta)} cols={list(meta.columns)}")

# ── Targets ──────────────────────────────────────────────────────────────────
TARGETS = [
    ("Splicing Index", "target_SI"),
    ("Hand Grip Strength (%)", "HGS_pp_avg"),
    ("Average Ankle Dorsiflexion (%)", "ADF_pp_avg"),
]
TARGET_ORDER = ("target_SI", "HGS_pp_avg", "ADF_pp_avg")
tgt_index = {name: i for i, name in enumerate(TARGET_ORDER)}

# ── Preprocessing grid ───────────────────────────────────────────────────────
PREPROCESS_GRID = [
    [],
    ["EMSC"],
    ["Normalization"],
    ["SNV"],
    ["Second Derivative"],
    ["EMSC","Normalization"],
    ["EMSC","SNV"],
    ["EMSC","Second Derivative"],
    ["Normalization","EMSC"],
    ["Normalization","SNV"],
    ["Normalization","Second Derivative"],
    ["SNV","EMSC"],
    ["SNV","Normalization"],
    ["SNV","Second Derivative"],
    ["Second Derivative","EMSC"],
    ["Second Derivative","Normalization"],
    ["Second Derivative","SNV"],
]

# ── Hyperparameter spaces (same as before) ───────────────────────────────────
PLS_COMPONENTS  = list(range(2, 19))
IPLS_COMPONENTS = [6]
IPLS_INTERVALS  = [15]

def generate_random_rf_params(num_iterations=100, seed=42):
    import random
    rng = random.Random(seed)
    grid = []
    for _ in range(num_iterations):
        grid.append({
            "n_estimators":     rng.choice([50, 100, 150, 160, 170, 180, 190]),
            "max_features":     "sqrt",
            "max_depth":        rng.choice([10, 15, 20, 25]),
            "min_samples_split":rng.choice([2, 10, 50, 100, 130, 140, 150, 160, 170, 180]),
            "min_samples_leaf": rng.choice([1, 10, 30, 40, 47, 48, 50, 55, 60]),
        })
    return grid

RF_GRID = generate_random_rf_params(num_iterations=100, seed=42)

ACFNN_GRID = [
    {"hidden_layer_sizes": (256, 128),       "activation": "relu", "alpha": 1e-4, "learning_rate_init": 1e-3, "batch_size": 64,  "max_iter": 200},
    {"hidden_layer_sizes": (512, 256),       "activation": "relu", "alpha": 1e-4, "learning_rate_init": 5e-4, "batch_size": 64,  "max_iter": 300},
    {"hidden_layer_sizes": (256, 256, 128),  "activation": "relu", "alpha": 1e-5, "learning_rate_init": 1e-3, "batch_size": 128, "max_iter": 300},
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def mse_1target(y_true_sample, y_pred_sample, col_idx: int) -> float:
    return float(_mse(y_true_sample[:, col_idx], y_pred_sample[:, col_idx]))

def _safe_methods_path(methods): return "+".join(m.replace(" ", "_") for m in methods) if methods else "none"

def _rf_tag(hp: dict) -> str:
    return (f"ne{hp['n_estimators']}_md{hp['max_depth']}_"
            f"mss{hp['min_samples_split']}_msl{hp['min_samples_leaf']}_mf{hp['max_features']}")

def _ac_tag(hp: dict) -> str:
    hls = "-".join(map(str, hp["hidden_layer_sizes"])) if isinstance(hp["hidden_layer_sizes"], (list, tuple)) else str(hp["hidden_layer_sizes"])
    return (f"hls{hls}_act{hp.get('activation','relu')}_bs{hp.get('batch_size', 'NA')}"
            f"_lr{hp.get('learning_rate_init','NA')}_wd{hp.get('alpha','0')}_mi{hp.get('max_iter','NA')}")

def _metrics_for_target(y_true_1d: np.ndarray, y_pred_1d: np.ndarray):
    rmse = float(np.sqrt(_mse(y_true_1d, y_pred_1d)))
    r2   = float(r2_score(y_true_1d, y_pred_1d))
    mae  = float(mean_absolute_error(y_true_1d, y_pred_1d))
    # safe MAPE (avoid divide by 0)
    denom = np.where(np.abs(y_true_1d) < 1e-8, np.nan, np.abs(y_true_1d))
    mape = float(np.nanmean(np.abs((y_pred_1d - y_true_1d) / denom)) * 100.0)
    lr = LinearRegression().fit(y_true_1d.reshape(-1,1), y_pred_1d.reshape(-1,1))
    slope = float(lr.coef_[0])
    intercept = float(lr.intercept_)
    return dict(rmse=rmse, r2=r2, mae=mae, mape=mape, slope=slope, intercept=intercept)

# ── Runner ───────────────────────────────────────────────────────────────────
rm = GlobalGroupedCV(all_spectra=all_spectra, meta=meta)

for nice_name, target_col in TARGETS:
    print(f"\n=== PER-TARGET selection for {nice_name} ({target_col}) ===")
    col_idx = tgt_index[target_col]

    # where to dump *all predictions* & *metrics* for this target
    allcombos_base = os.path.join(OUT_DIR, "all_combos", target_col)
    metrics_rows = []
    os.makedirs(allcombos_base, exist_ok=True)

    best = None  # (model_name, methods, hp_note, result_dict, score_for_this_target)

    for methods in PREPROCESS_GRID:
        # ============= PLS: global HP search over 2..18 (one pass to pick best for this target) =============
        pls_res = rm.run_pls_global(
            preprocess_methods=methods, n_components_list=PLS_COMPONENTS,
            n_splits=10, target_order=TARGET_ORDER
        )
        yts = pls_res["heldout_predictions"]["y_true_sample"]
        yps = pls_res["heldout_predictions"]["y_pred_sample"]
        score = mse_1target(yts, yps, col_idx)
        cand = ("pls", methods, {"n_components_list": PLS_COMPONENTS}, pls_res, score)
        best = cand if (best is None or score < best[4]) else best

        # also emit metrics & per-HP predictions (one file per HP)
        model_dir = os.path.join(allcombos_base, f"pls__{_safe_methods_path(methods)}")
        os.makedirs(model_dir, exist_ok=True)
        for nc in PLS_COMPONENTS:
            res1 = rm.run_pls_global(
                preprocess_methods=methods, n_components_list=[nc],
                n_splits=10, target_order=TARGET_ORDER
            )
            y_true = res1["heldout_predictions"]["y_true_sample"][:, col_idx]
            y_pred = res1["heldout_predictions"]["y_pred_sample"][:, col_idx]
            # predictions excel
            out_xlsx = os.path.join(model_dir, f"C={nc}.xlsx")
            dfp = pd.DataFrame({f"{target_col}__true": y_true, f"{target_col}__pred": y_pred})
            dfp.to_excel(out_xlsx, index=False)
            # metrics row
            cv_mse = list(res1["summary"]["hp_mse_curve"].values())[0]  # only one HP here
            mr = _metrics_for_target(y_true, y_pred)
            metrics_rows.append({
                "target": target_col, "model": "pls", "preprocessing": " + ".join(methods) if methods else "None",
                "hyperparams": json.dumps({"n_components": int(nc)}),
                "cv_mse": float(cv_mse), **mr
            })

        # ============= RF: global HP search over 100 seeded draws =============
        rf_res = rm.run_rf_global(
            preprocess_methods=methods, rf_param_list=RF_GRID,
            n_splits=10, target_order=TARGET_ORDER
        )
        yts = rf_res["heldout_predictions"]["y_true_sample"]
        yps = rf_res["heldout_predictions"]["y_pred_sample"]
        score = mse_1target(yts, yps, col_idx)
        cand = ("rf", methods, {"random_grid_len": len(RF_GRID)}, rf_res, score)
        best = cand if score < best[4] else best

        model_dir = os.path.join(allcombos_base, f"rf__{_safe_methods_path(methods)}")
        os.makedirs(model_dir, exist_ok=True)
        for hp in RF_GRID:
            res1 = rm.run_rf_global(
                preprocess_methods=methods, rf_param_list=[hp],
                n_splits=10, target_order=TARGET_ORDER
            )
            y_true = res1["heldout_predictions"]["y_true_sample"][:, col_idx]
            y_pred = res1["heldout_predictions"]["y_pred_sample"][:, col_idx]
            out_xlsx = os.path.join(model_dir, f"{_rf_tag(hp)}.xlsx")
            pd.DataFrame({f"{target_col}__true": y_true, f"{target_col}__pred": y_pred}).to_excel(out_xlsx, index=False)
            cv_mse = float(res1["summary"]["hp_mse_list"][0])
            mr = _metrics_for_target(y_true, y_pred)
            metrics_rows.append({
                "target": target_col, "model": "rf", "preprocessing": " + ".join(methods) if methods else "None",
                "hyperparams": json.dumps(hp), "cv_mse": cv_mse, **mr
            })

        # ============= AC-FNN: grid =============
        try:
            ac_res = rm.run_acfnn_global(
                preprocess_methods=methods, acfnn_param_list=ACFNN_GRID,
                n_splits=10, target_order=TARGET_ORDER
            )
            yts = ac_res["heldout_predictions"]["y_true_sample"]
            yps = ac_res["heldout_predictions"]["y_pred_sample"]
            score = mse_1target(yts, yps, col_idx)
            cand = ("acfnn", methods, {"grid_len": len(ACFNN_GRID)}, ac_res, score)
            best = cand if score < best[4] else best

            model_dir = os.path.join(allcombos_base, f"acfnn__{_safe_methods_path(methods)}")
            os.makedirs(model_dir, exist_ok=True)
            for hp in ACFNN_GRID:
                res1 = rm.run_acfnn_global(
                    preprocess_methods=methods, acfnn_param_list=[hp],
                    n_splits=10, target_order=TARGET_ORDER
                )
                y_true = res1["heldout_predictions"]["y_true_sample"][:, col_idx]
                y_pred = res1["heldout_predictions"]["y_pred_sample"][:, col_idx]
                out_xlsx = os.path.join(model_dir, f"{_ac_tag(hp)}.xlsx")
                pd.DataFrame({f"{target_col}__true": y_true, f"{target_col}__pred": y_pred}).to_excel(out_xlsx, index=False)
                cv_mse = float(res1["summary"]["hp_mse_list"][0])
                mr = _metrics_for_target(y_true, y_pred)
                metrics_rows.append({
                    "target": target_col, "model": "acfnn", "preprocessing": " + ".join(methods) if methods else "None",
                    "hyperparams": json.dumps(hp), "cv_mse": cv_mse, **mr
                })
        except AttributeError:
            print("AC-FNN runner not available; skipping.")

        # ============= iPLS: (C, I) =============
        ipls_res = rm.run_ipls_global(
            preprocess_methods=methods,
            n_components_list=IPLS_COMPONENTS,
            num_intervals_list=IPLS_INTERVALS,
            n_splits=10, target_order=TARGET_ORDER
        )
        yts = ipls_res["heldout_predictions"]["y_true_sample"]
        yps = ipls_res["heldout_predictions"]["y_pred_sample"]
        score = mse_1target(yts, yps, col_idx)
        cand = ("ipls", methods, {"n_components_list": IPLS_COMPONENTS, "num_intervals_list": IPLS_INTERVALS}, ipls_res, score)
        best = cand if score < best[4] else best

        model_dir = os.path.join(allcombos_base, f"ipls__{_safe_methods_path(methods)}")
        os.makedirs(model_dir, exist_ok=True)
        for c in IPLS_COMPONENTS:
            for I in IPLS_INTERVALS:
                res1 = rm.run_ipls_global(
                    preprocess_methods=methods,
                    n_components_list=[c], num_intervals_list=[I],
                    n_splits=10, target_order=TARGET_ORDER
                )
                y_true = res1["heldout_predictions"]["y_true_sample"][:, col_idx]
                y_pred = res1["heldout_predictions"]["y_pred_sample"][:, col_idx]
                out_xlsx = os.path.join(model_dir, f"C={c}__I={I}.xlsx")
                pd.DataFrame({f"{target_col}__true": y_true, f"{target_col}__pred": y_pred}).to_excel(out_xlsx, index=False)
                # hp_mse_table has one entry here
                cv_mse = float(list(res1["summary"]["hp_mse_table"].values())[0])
                mr = _metrics_for_target(y_true, y_pred)
                metrics_rows.append({
                    "target": target_col, "model": "ipls", "preprocessing": " + ".join(methods) if methods else "None",
                    "hyperparams": json.dumps({"n_components": int(c), "num_intervals": int(I)}),
                    "cv_mse": cv_mse, **mr
                })

    # === Emit artifacts for the winner (this target only) ===
    model_name, methods, hp_note, res, _ = best
    y_true_full = res["heldout_predictions"]["y_true_sample"]
    y_pred_full = res["heldout_predictions"]["y_pred_sample"]
    y_true_std = res["heldout_predictions"].get("y_true_sample_std")
    y_pred_std = res["heldout_predictions"].get("y_pred_sample_std")

    # keep only this target column (+ stds)
    y_true_1 = y_true_full[:, [col_idx]]
    y_pred_1 = y_pred_full[:, [col_idx]]
    y_true_1_std = None if y_true_std is None else y_true_std[:, [col_idx]]
    y_pred_1_std = None if y_pred_std is None else y_pred_std[:, [col_idx]]

    methods_label = " + ".join(methods) if methods else "No Preprocessing"
    methods_path  = _safe_methods_path(methods)
    title_suffix  = f"{model_name.upper()} | {methods_label} | HP: {res.get('chosen_hyperparams')}"

    tgt_dir = os.path.join(OUT_DIR, f"{target_col}", f"{model_name}__{methods_path}")
    os.makedirs(tgt_dir, exist_ok=True)

    # Parity plot (sample-level) with error bars
    parity_plot_sample_level(
        y_true_sample=y_true_1,
        y_pred_sample=y_pred_1,
        y_true_sample_std=y_true_1_std,
        y_pred_sample_std=y_pred_1_std,
        target_names=(nice_name,),
        out_dir=tgt_dir,
        fname_prefix="parity",
        title_suffix=title_suffix
    )

    # Save tidy predictions (winner)
    save_outer_predictions_excel(
        y_true_sample=y_true_1,
        y_pred_sample=y_pred_1,
        meta_kept=None,
        target_order=(target_col,),
        out_path=os.path.join(tgt_dir, f"{target_col}_outer_predictions.xlsx"),
        y_true_sample_std=y_true_1_std,
        y_pred_sample_std=y_pred_1_std,
    )

    # Write metrics Excel for this target (all combos)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_out = os.path.join(OUT_DIR, "metrics", f"{target_col}_metrics.xlsx")
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    metrics_df.to_excel(metrics_out, index=False)

    print(f"[BEST] {nice_name}: {model_name} + {methods_label} -> {res['summary']}")

print("\nDone.")
