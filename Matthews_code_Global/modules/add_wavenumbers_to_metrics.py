# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 14:37:18 2025

@author: spect
"""

# add_wavenumbers_to_metrics.py
# Standalone script: read existing ...__external_metrics_ALL.csv files, and
# append human-readable wavenumber columns (wn_start, wn_end, wn_center, wn_width)
# for iPLS intervals. Writes new CSVs (or overwrite with --overwrite).

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# --- import wn_tr from your existing loader
from modules.data_loader import load_data


def add_wn_cols(df: pd.DataFrame, wn: np.ndarray) -> pd.DataFrame:
    """
    Convert iPLS interval indices (a,b) → wavenumbers.
    NOTE: In your pipeline, X[:, a:b] uses b as EXCLUSIVE, so last included is (b-1).
    For non-iPLS rows (interval_* missing/NaN), new columns are NaN.
    """
    df = df.copy()
    if {"interval_a", "interval_b"}.issubset(df.columns):
        a = df["interval_a"].astype("Int64")  # nullable ints
        b = df["interval_b"].astype("Int64")

        # clip to valid bounds; keep NaN where present
        a_clip = a.clip(lower=0, upper=len(wn) - 1)
        b_clip = b.clip(lower=1, upper=len(wn))

        def idx_to_wn_start(i):
            return float(wn[int(i)]) if pd.notna(i) else np.nan

        def idx_to_wn_end(j):  # j = b (exclusive)
            return float(wn[int(j) - 1]) if pd.notna(j) else np.nan

        wn_start = a_clip.map(idx_to_wn_start)
        wn_end = b_clip.map(idx_to_wn_end)
        wn_center = (wn_start + wn_end) / 2.0
        wn_width = wn_end - wn_start

        df["wn_start"] = wn_start
        df["wn_end"] = wn_end
        df["wn_center"] = wn_center
        df["wn_width"] = wn_width
    else:
        # Ensure columns exist (all-NaN) so downstream code has consistent schema
        for c in ["wn_start", "wn_end", "wn_center", "wn_width"]:
            if c not in df.columns:
                df[c] = np.nan
    return df


def process_target(root: Path, tcol: str, wn: np.ndarray,
                   inpattern: str, out_suffix: str, overwrite: bool) -> None:
    """
    Read one target's metrics CSV, add wn columns, write output.
    """
    in_path = root / tcol / inpattern.format(tcol=tcol)
    if not in_path.exists():
        print(f"[WARN] Missing input CSV for {tcol}: {in_path}")
        return

    df = pd.read_csv(in_path)
    df_new = add_wn_cols(df, wn)

    if overwrite:
        out_path = in_path
    else:
        out_path = in_path.with_name(in_path.stem + out_suffix + in_path.suffix)

    df_new.to_csv(out_path, index=False)
    print(f"[OK] {tcol}: wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Append wavenumber columns to existing external metrics CSVs."
    )
    ap.add_argument("--root", required=True,
                    help="Root folder that contains per-target subfolders "
                         "(e.g., external_test_eval_ALL/target_SI).")
    ap.add_argument("--train-data", required=True,
                    help="TRAIN spectra folder (used only to load the wavenumber axis).")
    ap.add_argument("--train-meta", required=True,
                    help="TRAIN metadata CSV (passed to load_data).")
    ap.add_argument("--targets", nargs="*", default=["target_SI", "HGS_pp_avg", "ADF_pp_avg"],
                    help="Target folder names to process (must match subfolder names).")
    ap.add_argument("--inpattern", default="{tcol}__external_metrics_ALL.csv",
                    help="Filename pattern inside each target folder.")
    ap.add_argument("--out-suffix", default="_withWN",
                    help="Suffix for output CSV (before .csv). Use --overwrite to skip.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite the original CSV instead of writing a new file.")
    args = ap.parse_args()

    root = Path(args.root)

    # Load TRAIN wavenumber axis once (the grid is shared across all spectra)
    wn_tr, _, _, _ = load_data(
        data_dir=args.train_data,
        metadata_path=args.train_meta,
        include_types=("DM1", "Control"),
        return_filenames=False,
        strict=False,
        report_samples=0,
    )
    wn_tr = np.asarray(wn_tr)

    for tcol in args.targets:
        process_target(root, tcol, wn_tr, args.inpattern, args.out_suffix, args.overwrite)

    print("[DONE] All requested targets processed.")


if __name__ == "__main__":
    main()
