# -*- coding: utf-8 -*-
"""
main.py — per-target GLOBAL-HP 10-fold CV (PLS / RF / iPLS / AC-FNN)
Each target is tuned independently: every model × preprocessing × HP
is evaluated on that target only (no cross-eval across targets).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

from modules.ten_fold_cv_regression_global import GlobalGroupedCV
from modules.data_loader import load_data
from modules.plot_parity import parity_plot_sample_level, save_outer_predictions_excel

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
META_PATH = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"
OUT_DIR   = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Matthews_code_Global\results_global_hp_per_target"
os.makedirs(OUT_DIR, exist_ok=True)

# Which sample types to include (handled inside loader/regression as well)
INCLUDE_TYPES = ("DM1", "Control")

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

# ── Preprocessing chains to try ──────────────────────────────────────────────
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

# ── Model hyperparameter spaces ──────────────────────────────────────────────
PLS_COMPONENTS  = list(range(2, 19))     # 2..18
IPLS_COMPONENTS = list(range(2, 11))     # 2..10  ← change this
IPLS_INTERVALS  = [5, 10, 15]            #        ← and this

def generate_random_rf_params(num_iterations=100, seed=42):
    import random
    rng = random.Random(seed)
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

RF_GRID = generate_random_rf_params(num_iterations=100, seed=42)

# AC-FNN global HP grid (sklearn MLPRegressor stand-in unless you have a custom ACFNN)
ACFNN_GRID = [
    {"hidden_layer_sizes": (256, 128),        "activation": "relu", "alpha": 1e-4, "learning_rate_init": 1e-3, "batch_size": 64,  "max_iter": 200},
    {"hidden_layer_sizes": (512, 256),        "activation": "relu", "alpha": 1e-4, "learning_rate_init": 5e-4, "batch_size": 64,  "max_iter": 300},
    {"hidden_layer_sizes": (256, 256, 128),   "activation": "relu", "alpha": 1e-5, "learning_rate_init": 1e-3, "batch_size": 128, "max_iter": 300},
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def _metrics(y_true, y_pred):
    # 1-D arrays
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "rmse":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":    float(r2_score(y_true, y_pred)),
        "mae":   float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "evs":   float(explained_variance_score(y_true, y_pred)),
    }

def _safe_methods_path(methods):
    return "+".join(m.replace(" ", "_") for m in methods) if methods else "none"

def _rf_tag(hp: dict) -> str:
    return (f"ne{hp['n_estimators']}_md{hp['max_depth']}_"
            f"mss{hp['min_samples_split']}_msl{hp['min_samples_leaf']}_mf{hp['max_features']}")

def _ac_tag(hp: dict) -> str:
    hls = "-".join(map(str, hp["hidden_layer_sizes"])) if isinstance(hp["hidden_layer_sizes"], (list, tuple)) else str(hp["hidden_layer_sizes"])
    return (f"hls{hls}_act{hp.get('activation','relu')}_bs{hp.get('batch_size','NA')}"
            f"_lr{hp.get('learning_rate_init','NA')}_wd{hp.get('alpha','0')}_mi{hp.get('max_iter','NA')}")

# ── Runner (per-target, no cross-eval) ───────────────────────────────────────
rm = GlobalGroupedCV(all_spectra=all_spectra, meta=meta)

for nice_name, target_col in TARGETS:
    print(f"\n=== PER-TARGET search for {nice_name} ({target_col}) ===")
    TARGET_ORDER_ONE = (target_col,)   # <- single-target only
    COL_IDX = 0                        # first/only column

    # Where we dump files for this target
    tgt_root = os.path.join(OUT_DIR, target_col)
    os.makedirs(tgt_root, exist_ok=True)
    pred_root = os.path.join(tgt_root, "all_combos")  # per-HP predictions (one file per combo)
    os.makedirs(pred_root, exist_ok=True)

    metrics_rows = []   # rows for the consolidated metrics table
    best = None         # (score_rmse, model_name, methods, hp_str, res, paths)

    # Iterate all preprocessing chains
    for methods in PREPROCESS_GRID:
        mlabel = " + ".join(methods) if methods else "No Preprocessing"
        mpath  = _safe_methods_path(methods)

        # ---- PLS: evaluate every n_components ----
        model_dir = os.path.join(pred_root, f"pls__{mpath}")
        os.makedirs(model_dir, exist_ok=True)
        for nc in PLS_COMPONENTS:
            res = rm.run_pls_global(
                preprocess_methods=methods,
                n_components_list=[nc],
                n_splits=10,
                target_order=TARGET_ORDER_ONE,   # <- single target only
            )
            y_true = res["heldout_predictions"]["y_true_sample"][:, [COL_IDX]]
            y_pred = res["heldout_predictions"]["y_pred_sample"][:, [COL_IDX]]
            yts = y_true.ravel(); yps = y_pred.ravel()
            mets = _metrics(yts, yps)
            hp_str = f"C={nc}"

            # Save per-combo predictions
            out_xlsx = os.path.join(model_dir, f"{hp_str}.xlsx")
            pd.DataFrame({f"{target_col}__true": yts, f"{target_col}__pred": yps}).to_excel(out_xlsx, index=False)

            # Log metrics row
            metrics_rows.append({
                "target": target_col, "model": "pls", "preprocessing": mlabel, "hp": hp_str,
                **mets
            })

            # Track best by RMSE
            score = mets["rmse"]
            if (best is None) or (score < best[0]):
                best = (score, "pls", methods, hp_str, res, {"pred_path": out_xlsx})

        # ---- RF: evaluate every sampled HP ----
        model_dir = os.path.join(pred_root, f"rf__{mpath}")
        os.makedirs(model_dir, exist_ok=True)
        for hp in RF_GRID:
            res = rm.run_rf_global(
                preprocess_methods=methods,
                rf_param_list=[hp],
                n_splits=10,
                target_order=TARGET_ORDER_ONE,
            )
            y_true = res["heldout_predictions"]["y_true_sample"][:, [COL_IDX]]
            y_pred = res["heldout_predictions"]["y_pred_sample"][:, [COL_IDX]]
            yts = y_true.ravel(); yps = y_pred.ravel()
            mets = _metrics(yts, yps)
            hp_str = _rf_tag(hp)

            out_xlsx = os.path.join(model_dir, f"{hp_str}.xlsx")
            pd.DataFrame({f"{target_col}__true": yts, f"{target_col}__pred": yps}).to_excel(out_xlsx, index=False)

            metrics_rows.append({
                "target": target_col, "model": "rf", "preprocessing": mlabel, "hp": hp_str,
                **mets
            })

            score = mets["rmse"]
            if score < best[0]:
                best = (score, "rf", methods, hp_str, res, {"pred_path": out_xlsx})

        # ---- AC-FNN: evaluate every HP in the grid ----
        model_dir = os.path.join(pred_root, f"acfnn__{mpath}")
        os.makedirs(model_dir, exist_ok=True)
        for hp in ACFNN_GRID:
            res = rm.run_acfnn_global(
                preprocess_methods=methods,
                acfnn_param_list=[hp],
                n_splits=10,
                target_order=TARGET_ORDER_ONE,
            )
            y_true = res["heldout_predictions"]["y_true_sample"][:, [COL_IDX]]
            y_pred = res["heldout_predictions"]["y_pred_sample"][:, [COL_IDX]]
            yts = y_true.ravel(); yps = y_pred.ravel()
            mets = _metrics(yts, yps)
            hp_str = _ac_tag(hp)

            out_xlsx = os.path.join(model_dir, f"{hp_str}.xlsx")
            pd.DataFrame({f"{target_col}__true": yts, f"{target_col}__pred": yps}).to_excel(out_xlsx, index=False)

            metrics_rows.append({
                "target": target_col, "model": "acfnn", "preprocessing": mlabel, "hp": hp_str,
                **mets
            })

            score = mets["rmse"]
            if score < best[0]:
                best = (score, "acfnn", methods, hp_str, res, {"pred_path": out_xlsx})

        # ---- iPLS: evaluate every (components, intervals) combo ----
        model_dir = os.path.join(pred_root, f"ipls__{mpath}")
        os.makedirs(model_dir, exist_ok=True)
        for c in IPLS_COMPONENTS:
            for I in IPLS_INTERVALS:
                res = rm.run_ipls_global(
                    preprocess_methods=methods,
                    n_components_list=[c],
                    num_intervals_list=[I],
                    n_splits=10,
                    target_order=TARGET_ORDER_ONE,
                )
                y_true = res["heldout_predictions"]["y_true_sample"][:, [COL_IDX]]
                y_pred = res["heldout_predictions"]["y_pred_sample"][:, [COL_IDX]]
                yts = y_true.ravel(); yps = y_pred.ravel()
                mets = _metrics(yts, yps)
                hp_str = f"C={c}__I={I}"

                out_xlsx = os.path.join(model_dir, f"{hp_str}.xlsx")
                pd.DataFrame({f"{target_col}__true": yts, f"{target_col}__pred": yps}).to_excel(out_xlsx, index=False)

                metrics_rows.append({
                    "target": target_col, "model": "ipls", "preprocessing": mlabel, "hp": hp_str,
                    **mets
                })

                score = mets["rmse"]
                if score < best[0]:
                    best = (score, "ipls", methods, hp_str, res, {"pred_path": out_xlsx})

    # --- Write consolidated METRICS for this target (all combos) -------------
    metrics_df = pd.DataFrame(metrics_rows).sort_values(by=["rmse", "model", "preprocessing", "hp"]).reset_index(drop=True)
    metrics_path = os.path.join(tgt_root, f"{target_col}__metrics_all_combos.xlsx")
    metrics_df.to_excel(metrics_path, index=False)
    print(f"[METRICS] wrote {metrics_path}  ({len(metrics_df)} combos)")

    # --- Winner artifacts: parity plot (with error bars) + tidy predictions ---
    _, model_name, methods, hp_str, res_w, _paths = best
    y_true_full = res_w["heldout_predictions"]["y_true_sample"][:, [COL_IDX]]
    y_pred_full = res_w["heldout_predictions"]["y_pred_sample"][:, [COL_IDX]]
    y_true_std  = res_w["heldout_predictions"].get("y_true_sample_std")
    y_pred_std  = res_w["heldout_predictions"].get("y_pred_sample_std")
    y_true_std1 = None if y_true_std is None else y_true_std[:, [COL_IDX]]
    y_pred_std1 = None if y_pred_std is None else y_pred_std[:, [COL_IDX]]

    methods_label = " + ".join(methods) if methods else "No Preprocessing"
    methods_path  = _safe_methods_path(methods)
    title_suffix  = f"{model_name.upper()} | {methods_label} | HP: {hp_str}"

    # winner dir
    winner_dir = os.path.join(tgt_root, f"winner__{model_name}__{methods_path}")
    os.makedirs(winner_dir, exist_ok=True)

    # parity (error bars if available)
    parity_plot_sample_level(
        y_true_sample=y_true_full,
        y_pred_sample=y_pred_full,
        y_true_sample_std=y_true_std1,
        y_pred_sample_std=y_pred_std1,
        target_names=(nice_name,),
        out_dir=winner_dir,
        fname_prefix="parity",
        title_suffix=title_suffix
    )

    # tidy predictions for the winner (this target only)
    save_outer_predictions_excel(
        y_true_sample=y_true_full,
        y_pred_sample=y_pred_full,
        meta_kept=None,
        target_order=(target_col,),
        out_path=os.path.join(winner_dir, f"{target_col}__winner_predictions.xlsx"),
        y_true_sample_std=y_true_std1,
        y_pred_sample_std=y_pred_std1,
    )

    print(f"[BEST] {nice_name}: {model_name} + {methods_label} [{hp_str}] — RMSE={best[0]:.4f}")

print("\nDone.")
