# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 12:18:20 2025

@author: spect
"""

import glob
import os
import pickle

import pandas as pd

# adjust this to wherever your bayes_results lives
BASE = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\bayes_results"

all_folds  = []
all_global = []

for chain_dir in os.listdir(BASE):
    chain_path = os.path.join(BASE, chain_dir)
    pkl_path   = os.path.join(chain_path, f"results_{chain_dir}.pkl")
    if not os.path.isfile(pkl_path):
        continue

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # turn the two lists of dicts into DataFrames
    df_f = pd.DataFrame(data["folds"])
    df_g = pd.DataFrame(data["global"])

    # label with chain
    df_f["chain"] = chain_dir
    df_g["chain"] = chain_dir

    all_folds .append(df_f)
    all_global.append(df_g)

# concatenate everything
df_all_folds  = pd.concat(all_folds,  ignore_index=True)
df_all_global = pd.concat(all_global, ignore_index=True)

# ─── NEW: unpack the 9‐tuples into their own columns ────────────────────────
param_names = [
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "gamma",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
]

# ensemble‐fold hyperparams from best_hp
df_all_folds[param_names] = pd.DataFrame(
    df_all_folds["best_hp"].tolist(),
    index=df_all_folds.index
)

# global‐retrain hyperparams from mode_hp
df_all_global[param_names] = pd.DataFrame(
    df_all_global["mode_hp"].tolist(),
    index=df_all_global.index
)
# ────────────────────────────────────────────────────────────────────────────

# save to CSV (with hyperparams included)
df_all_folds .to_csv("all_chains_fold_results_with_hp.csv",  index=False)
df_all_global.to_csv("all_chains_global_results_with_hp.csv", index=False)

# also your original CSVs if you like
df_all_folds .to_csv("all_chains_fold_results.csv",  index=False)
df_all_global.to_csv("all_chains_global_results.csv", index=False)

print("Ensemble (per-fold) summary by chain:")
print(df_all_folds .groupby("chain")["fold_rmse"].mean().sort_values())

print("\nGlobal-HP (per-fold) summary by chain:")
print(df_all_global.groupby("chain")["global_rmse"].mean().sort_values())

# ─── Quick summaries of the hyperparameters themselves ──────────────────────
print("\nMost‐common global hyperparams per chain:")
for chain, grp in df_all_global.groupby("chain"):
    mode_hp = grp["mode_hp"].mode().iloc[0]
    print(f"  {chain:20s} → {mode_hp}")

print("\nAverage learning_rate by chain (global retrain):")
print(df_all_global.groupby("chain")["learning_rate"].mean().sort_values())
