#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_bayes_modular.py

Performs:
1) Outer LOO over 11 external test samples
2) Inner 5-fold GroupKFold with Bayesian optimization (optimize_xgb_with_cv)
3) Preprocessing per chain (fit on X_pool; transform held-out only)
4) Phase 1: per-fold tuned ensemble; Phase 2: global HPs (binned mode) re-run
5) Parity plots (ensemble vs global) saved per-chain
"""

import os
import numpy as np
import pandas as pd

from modules.data_loader import load_data, load_metadata
from modules.hyperparams import xgb_space
from modules.run_outer_loo import run_outer_loo
from modules.bayes_visualizations import plot_parity
import modules.bayes_visualizations as bvvis  # to set OUTPUT_DIR per chain


def load_external(train_dir, train_meta, test_dir, test_meta):
    """
    Returns:
        raw_tr (1731, 9*N_train)
        y_tr_meta (N_train,)
        raw_ex (1731, 9*N_test)
        y_ex_meta (N_test,)
        sample_ids_ex (list[str] with N_test items)
    """
    _, _, raw_tr, _ = load_data(train_dir, metadata_path=train_meta, return_filenames=True)
    train_meta_df = load_metadata(train_meta)
    y_tr_meta = train_meta_df['target_SI'].values

    _, _, raw_ex, _ = load_data(test_dir, metadata_path=test_meta, return_filenames=True)
    test_meta_df = load_metadata(test_meta)
    y_ex_meta = test_meta_df['target_SI'].values
    sample_ids_ex = test_meta_df.index.astype(str).tolist()

    # Basic consistency checks
    assert raw_ex.shape[1] % 9 == 0, "External set must have 9 spectra per sample."
    assert len(y_ex_meta) == raw_ex.shape[1] // 9, "Mismatch between test labels and spectra."

    return raw_tr, y_tr_meta, raw_ex, y_ex_meta, sample_ids_ex


if __name__ == '__main__':
    # ---- Paths (edit as needed) ----
    data_directory              = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
    meta_data_directory         = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"
    external_test_spectra_path  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data_test_updated"
    external_test_metadata_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata_test_updated.csv"
    output_dir                  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\bayes_results"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load external data ----
    raw_tr, y_tr_meta, raw_ex, y_ex_meta, sample_ids_ex = load_external(
        data_directory,
        meta_data_directory,
        external_test_spectra_path,
        external_test_metadata_path
    )
    print("✅ Data loaded. Running outer LOO across preprocessing chains...")

    # ---- Preprocessing chains ----
    preprocess_grid = [
        [],
    #     [], ['EMSC'], ['SNV'],
    #     ['Normalization'], ['Second Derivative'],
    #     ['EMSC', 'SNV'],
    #     ['EMSC', 'SNV', 'Second Derivative'],
    #     ['SNV', 'Second Derivative'],
    ]

    all_folds = []
    all_global = []

    for chain in preprocess_grid:
        chain_name = "+".join(chain) if chain else "none"
        base_out = os.path.join(output_dir, chain_name)
        os.makedirs(base_out, exist_ok=True)

        # Route chain-level plots to the chain folder
        bvvis.OUTPUT_DIR = base_out

        print(f"→ Chain: {chain_name}  (results in: {base_out})")
        folds, globals_ = run_outer_loo(
            raw_tr, y_tr_meta, raw_ex, y_ex_meta, sample_ids_ex,
            chain, xgb_space, base_out, n_calls=50
        )

        all_folds.extend(folds)
        all_global.extend(globals_)

        # ----- Chain-level parity summaries -----
        # Sort by fold to line up true/pred values
        df_folds = pd.DataFrame(folds).sort_values("fold")
        df_global = pd.DataFrame(globals_).sort_values("fold")

        # Build y_true from fold indices (ensures order matches)
        y_true = [y_ex_meta[i] for i in df_global["fold"]]
        
        # Ensemble (per-fold tuned)
        y_ens      = df_folds["fold_pred_mean"].tolist()
        y_ens_std  = df_folds["fold_pred_std"].tolist()
        
        # Global (train-only HPs; leak-free)
        y_glob     = df_global["global_pred_mean"].tolist()
        y_glob_std = df_global["global_pred_std"].tolist()
        
        # Save parity plots into chain folder (both with error bars)
        plot_parity(f"Ensemble_{chain_name}", y_true, y_ens,  stds=y_ens_std)
        plot_parity(f"Global_{chain_name}",   y_true, y_glob, stds=y_glob_std)


        # (Optional) save chain-level CSVs
        df_folds.to_csv(os.path.join(base_out, f"{chain_name}_ensemble_folds.csv"), index=False)
        df_global.to_csv(os.path.join(base_out, f"{chain_name}_global_folds.csv"), index=False)

    # ---- Combined summary across chains ----
    df_all_folds = pd.DataFrame(all_folds)
    df_all_global = pd.DataFrame(all_global)

    # Merge on keys present in both (fold + preproc); keeps both ensemble/global metrics
    df_both = df_all_folds.merge(df_all_global, on=["fold", "preproc"], suffixes=("_ens", "_glob"))
    df_all_folds.to_excel(os.path.join(output_dir, "loo_bayes_ensemble_all.xlsx"), index=False)
    df_all_global.to_excel(os.path.join(output_dir, "loo_bayes_global_all.xlsx"), index=False)
    df_both.to_excel(os.path.join(output_dir, "loo_bayes_comparison.xlsx"), index=False)

    print("✅ Done. Wrote combined Excel files and chain-level plots/CSVs.")
