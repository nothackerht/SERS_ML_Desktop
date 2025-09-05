# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:03:17 2025

@author: spect
"""

import os
import numpy as np
import pandas as pd
from modules.plot_parity import parity_plot_sample_level

# === adjust this to your results root ===
OUT_DIR = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Matthews_code_Global\results_global_hp_per_target"

# nice name, meta column
TARGETS = [
    ("Splicing Index", "target_SI"),
    ("Hand Grip Strength (%)", "HGS_pp_avg"),
    ("Average Ankle Dorsiflexion (%)", "ADF_pp_avg"),
]

def _safe_methods_path(methods_label: str) -> str:
    # reverse of what we used when saving folders
    if methods_label.lower().startswith("no"):
        return "none"
    parts = [p.strip().replace(" ", "_") for p in methods_label.split("+")]
    return "+".join(parts)

for nice, tcol in TARGETS:
    tgt_root = os.path.join(OUT_DIR, tcol)

    # 1) Read metrics table to recover the winning model + preprocessing + HP
    metrics_path = os.path.join(tgt_root, f"{tcol}__metrics_all_combos.xlsx")
    mets = pd.read_excel(metrics_path)
    best = mets.sort_values("rmse", ascending=True).iloc[0]
    model = str(best["model"])
    prep_label = str(best["preprocessing"])
    hp_str = str(best["hp"])

    # 2) Load the already-saved winner predictions (these include std columns if available)
    #    (this file was written by main.py)
    winner_dir = [d for d in os.listdir(tgt_root) if d.startswith("winner__")][0]
    pred_xlsx = os.path.join(tgt_root, winner_dir, f"{tcol}__winner_predictions.xlsx")
    df = pd.read_excel(pred_xlsx)

    y_true = df[f"{tcol}__true"].to_numpy().reshape(-1, 1)
    y_pred = df[f"{tcol}__pred"].to_numpy().reshape(-1, 1)

    # optional error bars (present for PLS/RF/iPLS/ACFNN in your pipeline)
    xt = df.get(f"{tcol}__true_std")
    yt = df.get(f"{tcol}__pred_std")
    xerr = None if xt is None else np.asarray(xt).reshape(-1, 1)
    yerr = None if yt is None else np.asarray(yt).reshape(-1, 1)

    # 3) Re-plot with wrapped subtitle so nothing gets cut off
    title_suffix = f"{model.upper()} | {prep_label} | HP: {hp_str}"
    out_dir = os.path.join(tgt_root, "winner_fixed")  # new folder so you keep the originals
    parity_plot_sample_level(
        y_true_sample=y_true,
        y_pred_sample=y_pred,
        y_true_sample_std=xerr,
        y_pred_sample_std=yerr,
        target_names=(nice,),
        out_dir=out_dir,
        fname_prefix="parity",
        dpi=600,
        title_suffix=title_suffix,
    )

    print(f"Replotted {nice} → {os.path.join(out_dir, 'parity_1.png')}")
