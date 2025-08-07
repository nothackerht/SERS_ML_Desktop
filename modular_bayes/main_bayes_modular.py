# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 12:51:22 2025

@author: spect
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_bayes_modular.py

Performs:
1. Outer LOO over 11 external test samples
2. Inner 5-fold GroupKFold with Bayesian optimization using optimize_xgb_with_cv()
3. Preprocessing + interval selection per chain
4. Fold-wise and global retraining + parity plot generation
"""

import os
import pickle
import numpy as np
import torch
import pandas as pd
from collections import Counter
from xgboost import XGBRegressor

from modules.data_loader import load_data, load_metadata
from modules.preprocessing import Preprocessing
from modules.hyperparams import xgb_space
from modules.bayes_visualizations import (
    plot_parity, plot_convergence_curve, plot_evaluations_scatter,
    plot_hyperparam_heatmap, plot_feature_importance, plot_residuals,
    plot_shap_summary
)
from modules.bayes_opt import optimize_xgb_with_cv
from modules.run_outer_loo import run_outer_loo
import modules.bayes_visualizations as bvvis  # just once at the top if not already

def load_external(train_dir, train_meta, test_dir, test_meta):
    _, _, raw_tr, _ = load_data(train_dir, metadata_path=train_meta, return_filenames=True)
    train_meta_df = load_metadata(train_meta)
    y_tr_meta = train_meta_df['target_SI'].values

    _, _, raw_ex, _ = load_data(test_dir, metadata_path=test_meta, return_filenames=True)
    test_meta_df = load_metadata(test_meta)
    y_ex_meta = test_meta_df['target_SI'].values
    sample_ids_ex = test_meta_df.index.astype(str).tolist()

    return raw_tr, y_tr_meta, raw_ex, y_ex_meta, sample_ids_ex


# Placeholder for run_outer_loo() — you’ll need to paste your full logic here,
# and replace the objective definition with a call to optimize_xgb_with_cv()

# Call this script with your defined preprocessing grid and base path
if __name__ == '__main__':
    data_directory              = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
    meta_data_directory         = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"
    external_test_spectra_path  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data_test_updated"
    external_test_metadata_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata_test_updated.csv"
    output_dir                  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\bayes_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load external data
    raw_tr, y_tr_meta, raw_ex, y_ex_meta, sample_ids_ex = load_external(
        data_directory,
        meta_data_directory,
        external_test_spectra_path,
        external_test_metadata_path
    )

    print("✅ Data loaded. Now ready to call run_outer_loo using optimize_xgb_with_cv()")
    preprocess_grid = [
        [], 
    ]
    # preprocess_grid = [
    #     [], ['EMSC'], ['SNV'],
    #     ['Normalization'], ['Second Derivative'],
    #     ['EMSC', 'SNV'], ['EMSC', 'SNV', 'Second Derivative'],
    #     ['SNV', 'Second Derivative'],
    # ]

    
    
    all_folds = []
    all_global = []
    
    for chain in preprocess_grid:
        base_out = os.path.join(output_dir, '+'.join(chain) if chain else "none")
        os.makedirs(base_out, exist_ok=True)
    
        folds, globals_ = run_outer_loo(
            raw_tr, y_tr_meta, raw_ex, y_ex_meta, sample_ids_ex,
            chain, xgb_space, base_out, n_calls=25
        )
    
        all_folds.extend(folds)
        all_global.extend(globals_)
        
        
        # Sort predictions by fold index to align them
        df_chain = pd.DataFrame(globals_)  # from global_records
        df_chain = df_chain.sort_values("fold")
        
        # Build parity data
        y_true = [y_ex_meta[i] for i in df_chain["fold"]]
        df_folds = pd.DataFrame(folds).sort_values("fold")
        y_ens = df_folds["fold_pred_mean"]
        y_std = df_folds["fold_pred_std"]

        y_glob = df_chain["global_pred_mean"]     # from global model
        
        # Set save location
        bvvis.OUTPUT_DIR = base_out  # saves into chain folder
        
        # Save plots
        plot_parity(f"Ensemble_{'+'.join(chain)}", y_true, y_ens, stds=y_std)
        plot_parity(f"Global_{'+'.join(chain)}",   y_true, y_glob)

    df_f = pd.DataFrame(all_folds)
    df_g = pd.DataFrame(all_global)
    df_both = df_f.merge(df_g, on=["fold", "preproc"])
    df_both.to_excel(os.path.join(output_dir, "loo_bayes_comparison.xlsx"), index=False)
