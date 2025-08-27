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
# ── Your modules ─────────────────────────────────────────────────────────────
from modules.data_loader import load_data                      # ✔ aligned spectra/meta  :contentReference[oaicite:2]{index=2}
from modules.ten_fold_CV_Fixed import run_nested_10fold_cv      # ← your new CV engine
# (Preprocessing is used inside 10_fold_CV_Fixed)              # ✔ API expects spectra as columns per sample  :contentReference[oaicite:3]{index=3}

# ── Paths (edit these) ──────────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
META_PATH  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"
OUT_DIR    = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\10_fold_results"

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
    [],                     # none
    ["EMSC"],
    ["Normalization"],
    ["SNV"],
    ["Second Derivative"],
    ["EMSC", "SNV"],
    ["SNV", "Second Derivative"],
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
all_results = []
for nice_name, target_col in TARGETS:
    print(f"\n=== Nested 10-fold CV: {nice_name} ({target_col}) ===")

    results_df, summary = run_nested_10fold_cv(
        X_all=all_spectra,          # shape (1731, 9N)  columns = spectra
        meta=meta,                  # aligned metadata (one row per sample)
        target_column=target_col,   # which column to predict
        preprocess_grid=PREPROCESS_GRID,
        model_list=MODEL_LIST,
        hyperparam_grids=HYPERPARAM_GRIDS,
        output_dir=os.path.join(OUT_DIR, f"{target_col}"),
        random_state=42,
        n_outer_splits=10,
        group_by_sample=True,       # group 9 spectra per patient/sample
        refit_pretreatment_each_fold=True,  # <— critical fix: refit preprocessing in every inner fold
        report_sample_level=True     # <— compute R²/MSE at sample level (9→1)
    )

    # Save per-target artifacts
    tgt_dir = os.path.join(OUT_DIR, f"{target_col}")
    os.makedirs(tgt_dir, exist_ok=True)
    results_path = os.path.join(tgt_dir, f"{target_col}_outer_predictions.xlsx")
    results_df.to_excel(results_path, index=False)

    print(f"[{target_col}] Sample-level metrics:\n{summary}")
    all_results.append((target_col, summary, results_path))

print("\nDone.")
