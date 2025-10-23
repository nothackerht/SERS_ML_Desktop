# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 12:57:51 2025

@author: spect

Paper launcher: runs 10-Fold CV, SPXY, and External Test across 4 scenarios
and saves outputs into your requested folders.

Usage (from "Modules for paper\\New modules"):
    python main.py                 # run everything
    python main.py --only cv       # just the 10-Fold CVs
    python main.py --only spxy     # just the SPXY runs
    python main.py --only external # just the 1–2 -> Box-3 evaluations
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent

# Paths to the three runnable modules (your module files in this same folder)
MOD_CV   = ROOT / "ten_fold_cv_regression_global.py"
MOD_SPXY = ROOT / "SPXY_eval_no_leakage.py"          # should accept CLI args (data/meta/include/out/results dirs)
MOD_EXT  = ROOT / "evaluate_external_test_all_combos.py"

# -------------------------- Canonical data & meta --------------------------
DATA_DIRS = {
    "data"         : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\data",                       # boxes 1–2
    "data_combined": r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\data_combined",             # boxes 1–3
    "data_test"    : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\data_test_updated",         # box 3 only
    "meta_12"      : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\y_metadata.csv",
    "meta_123"     : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\y_metadata_combined_in_order.csv",
    "meta_test"    : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\y_metadata_test_updated_in_order.csv",
}

# -------------------------- Output roots you requested --------------------------
OUT_CV = {
    "B12_with_controls" : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-2 with controls",
    "B12_no_controls"   : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-2 without controls",
    "B123_with_controls": r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-3 with controls",
    "B123_no_controls"  : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-3 without controls",
}
OUT_SPXY = {
    "B12_with_controls" : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\SPXY\box 1-2 with controls",
    "B12_no_controls"   : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\SPXY\box 1-2 without controls",
    "B123_with_controls": r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\SPXY\box 1-3 with controls",
    "B123_no_controls"  : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\SPXY\box 1-3 without controls",
}
# External test (train 1–2 -> test 3)
OUT_EXT = {
    "B12_with_controls" : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\External Test\box 3 with controls",
    "B12_no_controls"   : r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\External Test\box 3 without controls",
}

# -------------------------- Scenarios (data + meta + include_types) --------------------------
SCENARIOS = {
    # boxes 1–2
    "B12_with_controls": dict(
        data_dir=DATA_DIRS["data"], meta_path=DATA_DIRS["meta_12"],
        include_types="DM1,Control", tag="B12_with_controls"
    ),
    "B12_no_controls": dict(
        data_dir=DATA_DIRS["data"], meta_path=DATA_DIRS["meta_12"],
        include_types="DM1", tag="B12_no_controls"
    ),
    # boxes 1–3
    "B123_with_controls": dict(
        data_dir=DATA_DIRS["data_combined"], meta_path=DATA_DIRS["meta_123"],
        include_types="DM1,Control", tag="B123_with_controls"
    ),
    "B123_no_controls": dict(
        data_dir=DATA_DIRS["data_combined"], meta_path=DATA_DIRS["meta_123"],
        include_types="DM1", tag="B123_no_controls"
    ),
}

# -------------------------- Leaderboard roots for SPXY replay --------------------------
# For B12_* use "box1and2withcontrols" dirs; for B123_* use "COmbinedCVresultsnocontrol" dirs (to match your files)
SPXY_LEADERBOARDS = {
    "B12": {
        "ADF_pp_avg":  r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\ten_fold_cv_global_combined\box1and2withcontrols\ADF_pp_avg",
        "HGS_pp_avg":  r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\ten_fold_cv_global_combined\box1and2withcontrols\HGS_pp_avg",
        "target_SI":   r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\ten_fold_cv_global_combined\box1and2withcontrols\target_SI",
    },
    "B123": {
        "ADF_pp_avg":  r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\ten_fold_cv_global_combined\COmbinedCVresultsnocontrol\ADF_pp_avg",
        "HGS_pp_avg":  r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\ten_fold_cv_global_combined\COmbinedCVresultsnocontrol\HGS_pp_avg",
        "target_SI":   r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\ten_fold_cv_global_combined\COmbinedCVresultsnocontrol\target_SI",
    }
}

# -------------------------- Utils --------------------------
def _ensure_exists(path_str: str, kind: str):
    p = Path(path_str)
    if not p.exists():
        print(f"[WARN] {kind} not found: {path_str}")
    return p

def run_cmd(cmd: list[str]):
    print("\n[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, shell=False)
    if proc.returncode != 0:
        print(f"[ERROR] Command failed with code {proc.returncode}")
        sys.exit(proc.returncode)

def run_cv():
    print("\n==================== 10-FOLD CV (All Scenarios) ====================")
    _ensure_exists(MOD_CV, "module")
    for key, cfg in SCENARIOS.items():
        out_dir = OUT_CV[key]
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n--- CV: {key} ---")
        _ensure_exists(cfg["data_dir"], "data_dir")
        _ensure_exists(cfg["meta_path"], "meta_path")
        cmd = [
            sys.executable, str(MOD_CV),
            "--data_dir", cfg["data_dir"],
            "--meta_path", cfg["meta_path"],
            "--include_types", cfg["include_types"],
            "--out_dir", out_dir,
        ]
        run_cmd(cmd)

def run_spxy(repeats: int = 10, aggregate_selection: bool = True):
    print("\n==================== SPXY (All Scenarios) ====================")
    _ensure_exists(MOD_SPXY, "module")
    for key, cfg in SCENARIOS.items():
        out_dir = OUT_SPXY[key]
        os.makedirs(out_dir, exist_ok=True)

        # Choose leaderboard roots by scenario
        roots = SPXY_LEADERBOARDS["B12"] if key.startswith("B12") else SPXY_LEADERBOARDS["B123"]

        print(f"\n--- SPXY: {key} ---")
        _ensure_exists(cfg["data_dir"], "data_dir")
        _ensure_exists(cfg["meta_path"], "meta_path")
        for tname, rpath in roots.items():
            _ensure_exists(rpath, f"leaderboard[{tname}]")

        cmd = [
            sys.executable, str(MOD_SPXY),
            "--data_dir", cfg["data_dir"],
            "--meta_path", cfg["meta_path"],
            "--include_types", cfg["include_types"],
            "--out_dir", out_dir,
            "--results_dir_adf", roots["ADF_pp_avg"],
            "--results_dir_hgs", roots["HGS_pp_avg"],
            "--results_dir_si",  roots["target_SI"],
            "--spxy_repeats", str(int(repeats)),
            "--aggregate_selection_across_seeds", "1" if aggregate_selection else "0",
        ]
        run_cmd(cmd)

def run_external():
    print("\n==================== EXTERNAL TEST (Train 1–2 -> Test 3) ====================")
    _ensure_exists(MOD_EXT, "module")
    for key in ("B12_with_controls", "B12_no_controls"):
        cfg = SCENARIOS[key]
        out_dir = OUT_EXT[key]
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n--- External: {key} (train on boxes 1–2, test on box 3) ---")
        _ensure_exists(cfg["data_dir"], "train_data_dir")
        _ensure_exists(cfg["meta_path"], "train_meta_path")
        _ensure_exists(DATA_DIRS["data_test"], "test_data_dir")
        _ensure_exists(DATA_DIRS["meta_test"], "test_meta_path")

        cmd = [
            sys.executable, str(MOD_EXT),
            "--train_data_dir", cfg["data_dir"],
            "--train_meta_path", cfg["meta_path"],
            "--include_types", cfg["include_types"],
            "--test_data_dir", DATA_DIRS["data_test"],
            "--test_meta_path", DATA_DIRS["meta_test"],
            "--out_dir", out_dir,
            # (Optional) If you decide to restrict to a pre-locked set of combos, uncomment:
            # "--combos_from_cv", "<path-to-a-csv-list-of-combos>"
        ]
        run_cmd(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cv", "spxy", "external"], help="Run only one stage.")
    ap.add_argument("--spxy_repeats", type=int, default=10, help="Number of SPXY seeds to run.")
    ap.add_argument("--spxy_no_aggregate", action="store_true", help="Disable Option A aggregation in SPXY.")
    args = ap.parse_args()

    # Verify modules exist
    for m in (MOD_CV, MOD_SPXY, MOD_EXT):
        _ensure_exists(m, "module")

    if args.only == "cv":
        run_cv()
    elif args.only == "spxy":
        run_spxy(repeats=args.spxy_repeats, aggregate_selection=(not args.spxy_no_aggregate))
    elif args.only == "external":
        run_external()
    else:
        run_cv()
        run_spxy(repeats=args.spxy_repeats, aggregate_selection=(not args.spxy_no_aggregate))
        run_external()

if __name__ == "__main__":
    main()
