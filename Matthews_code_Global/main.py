# -*- coding: utf-8 -*-
"""
main.py — runner for GLOBAL-HP 10-fold CV (PLS / RF / iPLS)
Works with: modules.ten_fold_cv_regression_global.GlobalGroupedCV
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import mean_squared_error as _mse

from modules.ten_fold_cv_regression_global import GlobalGroupedCV
from modules.data_loader import load_data
from modules.plot_parity import parity_plot_sample_level, save_outer_predictions_excel
import pandas as pd
# ── Paths (edit as needed) ───────────────────────────────────────────────────
DATA_DIR  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
META_PATH = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"
OUT_DIR   = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Matthews_code_Global\results_global_hp"
os.makedirs(OUT_DIR, exist_ok=True)

# Which sample types to include
INCLUDE_TYPES = ("DM1", "Control")  # adjust if your metadata includes others you want

# ── Load data ────────────────────────────────────────────────────────────────
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

# ── Targets (display name, metadata column) ──────────────────────────────────
TARGETS = [
    ("Splicing Index", "target_SI"),
    ("Hand Grip Strength (%)", "HGS_pp_avg"),
    ("Average Ankle Dorsiflexion (%)", "ADF_pp_avg"),
]

# ── Preprocessing chains to try (names must match modules/preprocessing.py) ──
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

# ── Model hyperparameter spaces (global selection) ───────────────────────────
PLS_COMPONENTS = list(range(2, 19))     # 2..18
IPLS_COMPONENTS = [6]
IPLS_INTERVALS  = [15]

def generate_random_rf_params(num_iterations=100):
    import random
    grid = []
    for _ in range(num_iterations):
        grid.append({
            "n_estimators": random.choice([50, 100, 150, 160, 170, 180, 190]),
            "max_features": "sqrt",
            "max_depth": random.choice([10, 15, 20, 25]),
            "min_samples_split": random.choice([2, 10, 50, 100, 130, 140, 150, 160, 170, 180]),
            "min_samples_leaf": random.choice([1, 10, 30, 40, 47, 48, 50, 55, 60]),
        })
    return grid

# ── Helper: sample-level MSE for a single target column ──────────────────────
def mse_1target(y_true_sample, y_pred_sample, col_idx: int) -> float:
    return float(_mse(y_true_sample[:, col_idx], y_pred_sample[:, col_idx]))
# --- Helpers for filenames / tags ---
def _safe_methods_path(methods):
    return "+".join(m.replace(" ", "_") for m in methods) if methods else "none"

def _rf_tag(hp: dict) -> str:
    return (f"ne{hp['n_estimators']}_md{hp['max_depth']}_"
            f"mss{hp['min_samples_split']}_msl{hp['min_samples_leaf']}_mf{hp['max_features']}")

# ── Run: global-HP 10-fold CV per target, pick best model+prep by MSE ───────
rm = GlobalGroupedCV(all_spectra=all_spectra, meta=meta)
TARGET_ORDER = ("target_SI", "HGS_pp_avg", "ADF_pp_avg")
tgt_index = {name: i for i, name in enumerate(TARGET_ORDER)}

RF_GRID = generate_random_rf_params(num_iterations=100)

for nice_name, target_col in TARGETS:
    print(f"\n=== GLOBAL HP selection for {nice_name} ({target_col}) ===")
    # Where we’ll dump every model/preproc/HP Excel for this target
    allcombos_base = os.path.join(OUT_DIR, "all_combos", target_col)
    os.makedirs(allcombos_base, exist_ok=True)

    col_idx = tgt_index[target_col]
    best = None  # (model_name, methods, hp_note, result_dict, score)

    for methods in PREPROCESS_GRID:
        # ---- PLS (global HPs) ----
        pls_res = rm.run_pls_global(
            preprocess_methods=methods,
            n_components_list=PLS_COMPONENTS,
            n_splits=10,
            target_order=TARGET_ORDER,
        )
        yts = pls_res["heldout_predictions"]["y_true_sample"]
        yps = pls_res["heldout_predictions"]["y_pred_sample"]
        score = mse_1target(yts, yps, col_idx)
        cand = ("pls", methods, {"n_components_list": PLS_COMPONENTS}, pls_res, score)
        best = cand if (best is None or score < best[4]) else best
        # ---- SAVE: PLS predictions for EVERY n_components (one file per HP) ----
        model_dir = os.path.join(allcombos_base, f"pls__{_safe_methods_path(methods)}")
        os.makedirs(model_dir, exist_ok=True)

        for nc in PLS_COMPONENTS:
            pls_one = rm.run_pls_global(
                preprocess_methods=methods,
                n_components_list=[nc],  # single HP -> treated as the global HP
                n_splits=10,
                target_order=TARGET_ORDER,
            )
            y_true = pls_one["heldout_predictions"]["y_true_sample"]
            y_pred = pls_one["heldout_predictions"]["y_pred_sample"]

            out_xlsx = os.path.join(model_dir, f"C={nc}.xlsx")
            cols = {}
            for j, t in enumerate(TARGET_ORDER):
                cols[f"{t}__true"] = y_true[:, j]
                cols[f"{t}__pred"] = y_pred[:, j]
            pd.DataFrame(cols).to_excel(out_xlsx, index=False)

        # ---- RF (global HPs) ----
        rf_res = rm.run_rf_global(
            preprocess_methods=methods,
            rf_param_list=RF_GRID,
            n_splits=10,
            target_order=TARGET_ORDER,
        )
        yts = rf_res["heldout_predictions"]["y_true_sample"]
        yps = rf_res["heldout_predictions"]["y_pred_sample"]
        score = mse_1target(yts, yps, col_idx)
        cand = ("rf", methods, {"random_grid_len": len(RF_GRID)}, rf_res, score)
        best = cand if score < best[4] else best
        # ---- SAVE: RF predictions for EVERY hyperparameter dict (one file per HP) ----
        model_dir = os.path.join(allcombos_base, f"rf__{_safe_methods_path(methods)}")
        os.makedirs(model_dir, exist_ok=True)

        for hp in RF_GRID:
            rf_one = rm.run_rf_global(
                preprocess_methods=methods,
                rf_param_list=[hp],   # single HP -> treated as the global HP
                n_splits=10,
                target_order=TARGET_ORDER,
            )
            y_true = rf_one["heldout_predictions"]["y_true_sample"]
            y_pred = rf_one["heldout_predictions"]["y_pred_sample"]

            out_xlsx = os.path.join(model_dir, f"{_rf_tag(hp)}.xlsx")
            cols = {}
            for j, t in enumerate(TARGET_ORDER):
                cols[f"{t}__true"] = y_true[:, j]
                cols[f"{t}__pred"] = y_pred[:, j]
            pd.DataFrame(cols).to_excel(out_xlsx, index=False)

        # ---- iPLS (global HPs; interval re-selected per fold from train only) ----
        ipls_res = rm.run_ipls_global(
            preprocess_methods=methods,
            n_components_list=IPLS_COMPONENTS,
            num_intervals_list=IPLS_INTERVALS,
            n_splits=10,
            target_order=TARGET_ORDER,
        )
        yts = ipls_res["heldout_predictions"]["y_true_sample"]
        yps = ipls_res["heldout_predictions"]["y_pred_sample"]
        score = mse_1target(yts, yps, col_idx)
        cand = ("ipls", methods, {"n_components_list": IPLS_COMPONENTS, "num_intervals_list": IPLS_INTERVALS}, ipls_res, score)
        best = cand if score < best[4] else best
        # ---- SAVE: iPLS predictions for EVERY (components, intervals) combo ----
        model_dir = os.path.join(allcombos_base, f"ipls__{_safe_methods_path(methods)}")
        os.makedirs(model_dir, exist_ok=True)

        for c in IPLS_COMPONENTS:
            for I in IPLS_INTERVALS:
                ipls_one = rm.run_ipls_global(
                    preprocess_methods=methods,
                    n_components_list=[c],      # single HP set -> global HPs
                    num_intervals_list=[I],
                    n_splits=10,
                    target_order=TARGET_ORDER,
                )
                y_true = ipls_one["heldout_predictions"]["y_true_sample"]
                y_pred = ipls_one["heldout_predictions"]["y_pred_sample"]

                out_xlsx = os.path.join(model_dir, f"C={c}__I={I}.xlsx")
                cols = {}
                for j, t in enumerate(TARGET_ORDER):
                    cols[f"{t}__true"] = y_true[:, j]
                    cols[f"{t}__pred"] = y_pred[:, j]
                pd.DataFrame(cols).to_excel(out_xlsx, index=False)

    # --- Emit artifacts for the winner (per target) ---
    model_name, methods, hp_note, res, _ = best
    y_true_full = res["heldout_predictions"]["y_true_sample"]
    y_pred_full = res["heldout_predictions"]["y_pred_sample"]
    # NEW: stds for error bars (present in global code's outputs)
    y_true_std = res["heldout_predictions"].get("y_true_sample_std")
    y_pred_std = res["heldout_predictions"].get("y_pred_sample_std")

    # keep only this target column for parity/export (+ stds for error bars)
    y_true_1 = y_true_full[:, [col_idx]]
    y_pred_1 = y_pred_full[:, [col_idx]]
    y_true_1_std = None if y_true_std is None else y_true_std[:, [col_idx]]
    y_pred_1_std = None if y_pred_std is None else y_pred_std[:, [col_idx]]


    methods_label = " + ".join(methods) if methods else "No Preprocessing"
    methods_path  = "+".join(m.replace(" ", "_") for m in methods) if methods else "none"
    # NEW: show model, preprocessing, and chosen HPs in plot title
    title_suffix = f"{model_name.upper()} | {methods_label} | HP: {res.get('chosen_hyperparams')}"

    tgt_dir = os.path.join(OUT_DIR, f"{target_col}", f"{model_name}__{methods_path}")
    os.makedirs(tgt_dir, exist_ok=True)

    # Parity plot (sample-level) for this target (with error bars + title)
    parity_plot_sample_level(
        y_true_sample=y_true_1,
        y_pred_sample=y_pred_1,
        y_true_sample_std=y_true_1_std,   # NEW: x-error bars
        y_pred_sample_std=y_pred_1_std,   # NEW: y-error bars
        target_names=(nice_name,),
        out_dir=tgt_dir,
        fname_prefix="parity",
        title_suffix=title_suffix         # NEW: show model/preproc/HP in title
    )

    # Save tidy predictions for this target
    save_outer_predictions_excel(
        y_true_sample=y_true_1,
        y_pred_sample=y_pred_1,
        meta_kept=None,
        target_order=(target_col,),
        out_path=os.path.join(tgt_dir, f"{target_col}_outer_predictions.xlsx"),
    )

    print(f"[BEST] {nice_name}: {model_name} + {methods_label} -> {res['summary']}")

print("\nDone.")
