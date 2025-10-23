# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 16:09:01 2025

@author: spect
"""

# summarize_external_correlation.py
# Standalone: computes Pearson correlation (raw & calibrated) for each target
# by reading the outputs produced by evaluate_external_test_linearized.py

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr
except Exception:
    pearsonr = None


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (r, p). Falls back to numpy corrcoef if SciPy isn't available."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    # Guardrails: need variance and at least 3 points
    if y_true.size < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan"), float("nan")

    if pearsonr is not None:
        r, p = pearsonr(y_true, y_pred)
        return float(r), float(p)

    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r), float("nan")


def load_metrics_json(target_dir: Path, tcol: str):
    mpath = target_dir / f"{tcol}__external_metrics.json"
    if not mpath.exists():
        return None
    with mpath.open("r") as f:
        return json.load(f)


def summarize_target(root: Path, tcol: str, sheet: str = "calibrated_preds"):
    tdir = root / tcol
    xlsx = tdir / f"{tcol}__external_predictions.xlsx"

    if not xlsx.exists():
        print(f"[WARN] Missing predictions file for {tcol}: {xlsx}")
        return

    # Read calibrated sheet (it includes raw + calibrated columns)
    try:
        df = pd.read_excel(xlsx, sheet_name=sheet)
    except Exception as e:
        print(f"[WARN] Could not read sheet '{sheet}' for {tcol} ({e}).")
        return

    # Column names produced by the evaluator
    y_true_col = f"{tcol}__y_true"
    y_raw_col = f"{tcol}__y_pred_raw"
    y_cal_col = f"{tcol}__y_pred_calibrated"

    missing = [c for c in [y_true_col, y_raw_col, y_cal_col] if c not in df.columns]
    if missing:
        print(f"[WARN] Expected columns not found for {tcol}: {missing}")
        return

    y_true = df[y_true_col].to_numpy()
    y_raw = df[y_raw_col].to_numpy()
    y_cal = df[y_cal_col].to_numpy()

    # If calibrants mode was used, there may be an "is_calibrant" flag; if so,
    # compute calibrated r only on non-calibrants (evaluation set) to match metrics.
    if "is_calibrant" in df.columns and df["is_calibrant"].dtype == bool:
        eval_mask = ~df["is_calibrant"].to_numpy()
    else:
        eval_mask = np.ones_like(y_true, dtype=bool)

    r_raw, p_raw = safe_pearson(y_true, y_raw)
    r_cal, p_cal = safe_pearson(y_true[eval_mask], y_cal[eval_mask])

    # Load RMSE/R² from metrics JSON if available
    mj = load_metrics_json(tdir, tcol)
    if mj is not None:
        mr = mj.get("metrics_sample_level_raw", {})
        mc = mj.get("metrics_sample_level_calibrated", {})
        rmse_raw = mr.get("rmse", float("nan"))
        r2_raw = mr.get("r2", float("nan"))
        rmse_cal = mc.get("rmse", float("nan"))
        r2_cal = mc.get("r2", float("nan"))
    else:
        rmse_raw = r2_raw = rmse_cal = r2_cal = float("nan")

    print(f"\n=== {tcol} ===")
    print(f"Samples (total): {y_true.size} | Evaluated (calibrated): {int(eval_mask.sum())}")
    print(f"RAW:         RMSE={rmse_raw:.4f}  R²={r2_raw:.4f}  Pearson r={r_raw:.3f}  p={p_raw:.3g}")
    print(f"CALIBRATED:  RMSE={rmse_cal:.4f}  R²={r2_cal:.4f}  Pearson r={r_cal:.3f}  p={p_cal:.3g}")

    # Optional: write a one-line CSV summary per target
    summary_path = tdir / f"{tcol}__pearson_summary.csv"
    pd.DataFrame([{
        "target": tcol,
        "n_total": int(y_true.size),
        "n_eval_calibrated": int(eval_mask.sum()),
        "raw_rmse": rmse_raw, "raw_r2": r2_raw, "raw_pearson_r": r_raw, "raw_pearson_p": p_raw,
        "cal_rmse": rmse_cal, "cal_r2": r2_cal, "cal_pearson_r": r_cal, "cal_pearson_p": p_cal,
    }]).to_csv(summary_path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Summarize Pearson correlation for external predictions.")
    ap.add_argument("--root", type=str, required=True,
                    help="Root directory produced by evaluate_external_test_linearized.py "
                         "(contains subfolders target_SI, HGS_pp_avg, ADF_pp_avg).")
    # Allow custom target list, default to your three
    ap.add_argument("--targets", type=str, nargs="*", default=["target_SI", "HGS_pp_avg", "ADF_pp_avg"],
                    help="List of target column folders to summarize.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    print(f"[INFO] Scanning: {root}")
    for tcol in args.targets:
        summarize_target(root, tcol)

    # Optional: aggregate all per-target summaries into a single CSV at the root
    rows = []
    for tcol in args.targets:
        p = root / tcol / f"{tcol}__pearson_summary.csv"
        if p.exists():
            rows.append(pd.read_csv(p))
    if rows:
        agg = pd.concat(rows, ignore_index=True)
        agg.to_csv(root / "external_pearson_summary_ALL.csv", index=False)
        print(f"\n[OK] Wrote aggregate summary → {root / 'external_pearson_summary_ALL.csv'}")


if __name__ == "__main__":
    import sys
    # Default root so it works in Spyder/runfile without args
    DEFAULT_ROOT = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\external_test_eval_linearized"
    if len(sys.argv) == 1:
        # Simulate: --root DEFAULT_ROOT
        sys.argv += ["--root", DEFAULT_ROOT]
    main()
