# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 15:37:12 2025

@author: spect
"""

# main.py
# Minimal runner for corrected grouped nested 10-fold CV
# Uses modules: data_loader.py, preprocessing.py, and your 10_fold_CV_Fixed.py

import os
import numpy as np
import pandas as pd
from modules.ten_fold_CV_Fixed import NestedGroupedCV
from modules.data_loader import load_data
from modules.plot_parity import parity_plot_sample_level, save_outer_predictions_excel
from sklearn.metrics import mean_squared_error as _mse
# ── Your modules ─────────────────────────────────────────────────────────────
                   # ✔ aligned spectra/meta  :contentReference[oaicite:2]{index=2}

# (Preprocessing is used inside 10_fold_CV_Fixed)              # ✔ API expects spectra as columns per sample  :contentReference[oaicite:3]{index=3}

# ── Paths (edit these) ──────────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
META_PATH  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"
OUT_DIR    = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Matthews_code_fixed\Matthew_approach_fixed_results"

os.makedirs(OUT_DIR, exist_ok=True)

# Include DM1 ± controls (toggle as needed)
INCLUDE_TYPES = ("DM1", "Control", "AdCo", "AdCo2")  # trim to what exists in your metadata

# ── Load data (aligned & validated) ─────────────────────────────────────────
wavenumbers, averaged_spectra, all_spectra, filenames_per_column, meta = load_data(
    data_dir=DATA_DIR,
    metadata_path=META_PATH,
    include_types=INCLUDE_TYPES,
    return_filenames=True,
    strict=False,
    report_samples=5
)

print(f"[SHAPES] wav: {wavenumbers.shape} | averaged: {averaged_spectra.shape} | all: {all_spectra.shape}")
print(f"[META] rows={len(meta)} cols={list(meta.columns)}")

# ── Targets to evaluate (pick one or many) ──────────────────────────────────
# Valid column names must exist in your metadata (e.g., 'target_SI', 'ADF_pp_avg', 'HGS_pp_avg')
TARGETS = [
    ("Splicing Index", "target_SI"),
    ("Hand Grip Strength (%)", "HGS_pp_avg"),
    ("Average Ankle Dorsiflexion (%)", "ADF_pp_avg"),
]

# ── Preprocessing chains to try (do not add new methods) ────────────────────
# These names must match your Preprocessing API inside modules/preprocessing.py
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


# ── Model space (unchanged models/hyperparams) ──────────────────────────────
# Keep the same models/params you used previously; the 10_fold_CV_Fixed module
# should honor them and *refit preprocessing inside each inner fold*.
MODEL_LIST = ["pls", "rf", "ipls"]     # example labels expected by your CV module

HYPERPARAM_GRIDS = {
    "pls":  {"n_components_list": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]},
    "rf":   {"num_random_draws": 100},  # uses your RF sampler inside the module
    "ipls": {"n_components_list": [6], "num_intervals_list": [15]},
}

# ── Run nested 10-fold CV per target ────────────────────────────────────────

rm = NestedGroupedCV(all_spectra=all_spectra, meta=meta)

TARGET_ORDER = ("target_SI", "HGS_pp_avg", "ADF_pp_avg")
PGRID = PREPROCESS_GRID  # reuse the list you defined above
OUT_BASE = OUT_DIR



def mse_1target(y_true_full, y_pred_full, col_idx: int) -> float:
    """MSE for a single target column (sample-level)."""
    return float(_mse(y_true_full[:, col_idx], y_pred_full[:, col_idx]))

# Map target name to its column index in TARGET_ORDER
tgt_index = {name: i for i, name in enumerate(TARGET_ORDER)}

for nice_name, target_col in TARGETS:
    print(f"\n=== Selecting best combo for {nice_name} ({target_col}) ===")
    best = None  # (model_name, methods, hp_note_dict, result_dict, score)

    col_idx = tgt_index[target_col]

    for methods in PGRID:
        # ---- PLS ----
        pls_res = rm.run_pls_nested(
            preprocess_methods=methods,
            n_components_list=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
            n_splits_outer=10,
            n_splits_inner=5,
            target_order=TARGET_ORDER
        )
        yts = pls_res["heldout_predictions"]["y_true_sample"]
        yps = pls_res["heldout_predictions"]["y_pred_sample"]
        score = mse_1target(yts, yps, col_idx)
        cand = ("pls", methods, {"n_components_choices": "2..18"}, pls_res, score)
        best = cand if (best is None or score < best[4]) else best

        # ---- RF ----
        rf_grid = rm.generate_random_rf_params(num_iterations=100)
        rf_res = rm.run_rf_nested(
            preprocess_methods=methods,
            rf_param_list=rf_grid,
            n_splits_outer=10,
            n_splits_inner=5,
            target_order=TARGET_ORDER
        )
        yts = rf_res["heldout_predictions"]["y_true_sample"]
        yps = rf_res["heldout_predictions"]["y_pred_sample"]
        score = mse_1target(yts, yps, col_idx)
        cand = ("rf", methods, {"random_grid_len": len(rf_grid)}, rf_res, score)
        best = cand if score < best[4] else best

        # ---- iPLS ----
        ipls_res = rm.run_ipls_nested(
            preprocess_methods=methods,
            n_components_list=[6],         # match your prior search
            num_intervals_list=[15],
            n_splits_outer=10,
            n_splits_inner=5,
            target_order=TARGET_ORDER
        )
        yts = ipls_res["heldout_predictions"]["y_true_sample"]
        yps = ipls_res["heldout_predictions"]["y_pred_sample"]
        score = mse_1target(yts, yps, col_idx)
        cand = ("ipls", methods, {"n_components": [6], "num_intervals": [15]}, ipls_res, score)
        best = cand if score < best[4] else best

    # --- Emit artifacts for the winner (per target) ---
    model_name, methods, hp_note, res, _ = best
    y_true_full = res["heldout_predictions"]["y_true_sample"]
    y_pred_full = res["heldout_predictions"]["y_pred_sample"]

    # Keep only the selected target column for plotting/export (shape (N,1))
    y_true_1 = y_true_full[:, [col_idx]]
    y_pred_1 = y_pred_full[:, [col_idx]]

    methods_label = " + ".join(methods) if methods else "No Preprocessing"
    methods_path  = "+".join(m.replace(" ", "_") for m in methods) if methods else "none"


    tgt_dir = os.path.join(OUT_BASE, f"{target_col}", f"{model_name}__{methods_path}")
    os.makedirs(tgt_dir, exist_ok=True)

    # Parity plot for this target only
    parity_plot_sample_level(
        y_true_sample=y_true_1,
        y_pred_sample=y_pred_1,
        target_names=(nice_name,),
        out_dir=tgt_dir,
        fname_prefix="parity"
    )

    # Save tidy predictions (this target only)
    save_outer_predictions_excel(
        y_true_sample=y_true_1,
        y_pred_sample=y_pred_1,
        meta_kept=None,  # optional: pass filtered meta if you want it merged
        target_order=(target_col,),
        out_path=os.path.join(tgt_dir, f"{target_col}_outer_predictions.xlsx")
    )

    print(f"[BEST] {nice_name}: {model_name} + {methods_label} -> {res['summary']}")

print("\nDone.")
