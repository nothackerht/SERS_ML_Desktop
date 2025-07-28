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

# save to CSV or inspect in pandas
df_all_folds .to_csv("all_chains_fold_results.csv",  index=False)
df_all_global.to_csv("all_chains_global_results.csv", index=False)

print("Ensemble (per-fold) summary by chain:")
print(df_all_folds .groupby("chain")["fold_rmse"].mean().sort_values())

print("\nGlobal-HP (per-fold) summary by chain:")
print(df_all_global.groupby("chain")["global_rmse"].mean().sort_values())
