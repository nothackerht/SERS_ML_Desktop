# -*- coding: utf-8 -*-
"""
main_PCA_Classification.py

Runs:
  • Classification (PLS-DA and SVM with LOSOCV) on the *combined* dataset
  • PCA analyses (train-only + train→test projection overlays, with optional SI coloring)
    on the *separate* train and test sets.

Outputs are saved to:
  • cls_output_dir  (classification)
  • pca_output_dir  (PCA)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: GPU info (not required for sklearn, but handy to see what's available)
try:
    import torch
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
except Exception:
    pass

# ── Project imports ────────────────────────────────────────────────────────────
from config import USE_CONTROLS, CONTROL_LABELS  # toggle controls globally
from modules.data_loader import load_data        # robust spectra+metadata loader
from modules.classification import Classifier    # PLS-DA & SVM LOSOCV wrappers
from modules.unsupervised_clustering import UnsupervisedClustering
from modules.preprocessing import Preprocessing
from sklearn.decomposition import PCA  # for train→test projections

# ── User paths: adjust if needed ───────────────────────────────────────────────
# COMBINED data (used ONLY for classification)
classification_data_dir_combined  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\data_combined"
classification_meta_path_combined = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata_combined_in_order.csv"

# Separate TRAIN data (used by PCA)
train_data_dir  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
train_meta_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"

# Separate TEST data (used by PCA)
test_data_dir  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data_test_updated"
test_meta_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata_test_updated.csv"

# Output folders
cls_output_dir = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Classification_Results"
pca_output_dir = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\PCA_Results"
os.makedirs(cls_output_dir, exist_ok=True)
os.makedirs(pca_output_dir, exist_ok=True)

# ── Classification settings ───────────────────────────────────────────────────
# PLS-DA
plsda_preprocess   = ['SNV', 'Normalization']
plsda_n_components = 15

# SVM
svm_preprocess = ['SNV']
svm_C          = 1.0
svm_kernel     = 'linear'  # 'linear' or 'rbf', etc.

# ── PCA suite (train/test overlays, severity coloring) ─────────────────────────
def run_pca_suite(
    train_data_dir: str,
    train_meta_path: str,
    test_data_dir: str,
    test_meta_path: str,
    include_types: tuple,
    output_dir: str,
):
    """
    Runs PCA on train and test:
      • Train-only PCA plots (PC1 vs PC2) across several preprocessing chains
      • Train-fitted PCA projected onto the test set (overlay/colored)
      • Optional severity/target-color overlays (SI) if present in metadata
    Saves figures to `output_dir`.
    """

    # ---- Load TRAIN and TEST with current loader (returns 5 items) ----
    wavs_train, _, spectra_train, train_filenames, meta_train = load_data(
        data_dir=train_data_dir,
        metadata_path=train_meta_path,
        include_types=include_types,
        return_filenames=True,
        strict=False,
        report_samples=5,
    )
    wavs_test, _, spectra_test, test_filenames, meta_test = load_data(
        data_dir=test_data_dir,
        metadata_path=test_meta_path,
        include_types=include_types,
        return_filenames=True,
        strict=False,
        report_samples=5,
    )

    # ---- Preprocessing combos to iterate ----
    preprocessing_combos = [
        [],  # No preprocessing
        ['Normalization'],
        ['SNV'],
        ['EMSC'],
        ['Second Derivative'],
        ['Normalization', 'SNV'],
        ['EMSC', 'Second Derivative'],
    ]

    # ---- Train-only PCA via your UnsupervisedClustering helpers ----
    clustering_train = UnsupervisedClustering(meta_train, spectra_train, output_dir)

    for methods in preprocessing_combos:
        name_part = '__'.join(methods) if methods else 'no_preprocessing'

        # 1) Train-only PCA (PC1 vs PC2)
        clustering_train.perform_pca_and_plot(
            x_array=spectra_train,
            plot_name=f"pca_train_only_pc1_vs_pc2_{name_part}",
            pc_axes=(1, 2),
            save_plot=True,
            preprocessing=methods
        )

        # 1b) Train-only PCA colored by Splicing Index (if present)
        if 'target_SI' in meta_train.columns:
            clustering_train.perform_pca_and_plot_with_severity(
                x_array=spectra_train,
                plot_name=f"pca_train_only_pc1_vs_pc2_severity_{name_part}",
                pc_axes=(1, 2),
                save_plot=True,
                preprocessing=methods
            )

        # 2) Train→Test projection using sklearn PCA directly
        #    (fit on preprocessed TRAIN; transform both TRAIN and TEST)
        X_train_prep = (
            Preprocessing(spectra_train).preprocess(methods=methods)
            if methods else spectra_train
        )
        X_test_prep = (
            Preprocessing(spectra_test).preprocess(methods=methods)
            if methods else spectra_test
        )

        # sklearn expects samples×features
        Xtr = X_train_prep.T
        Xte = X_test_prep.T

        pca = PCA(n_components=3)
        train_scores = pca.fit_transform(Xtr)
        test_scores  = pca.transform(Xte)

        # Build metadata repeated 9× (one row per spectrum)
        meta_train_rep = pd.DataFrame(np.repeat(meta_train.values, 9, axis=0), columns=meta_train.columns)
        meta_test_rep  = pd.DataFrame(np.repeat(meta_test.values,  9, axis=0), columns=meta_test.columns)
        meta_train_rep['Set'] = 'Train'
        meta_test_rep['Set']  = 'Test'

        # Combine and assign PCA scores
        combined_meta = pd.concat([meta_train_rep, meta_test_rep], ignore_index=True)
        combined_meta['PC1'] = np.concatenate([train_scores[:, 0], test_scores[:, 0]])
        combined_meta['PC2'] = np.concatenate([train_scores[:, 1], test_scores[:, 1]])

        # 3) Overlay: always show TRAIN (circles) + TEST (large stars)
        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        
        m_tr = combined_meta['Set'] == 'Train'
        m_te = combined_meta['Set'] == 'Test'
        
        if 'target_SI' in combined_meta.columns:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
        
            fig, ax = plt.subplots(figsize=(12, 8))
            cmap = 'viridis'
        
            # Train = circles (small)
            ax.scatter(
                combined_meta.loc[m_tr, 'PC1'],
                combined_meta.loc[m_tr, 'PC2'],
                c=combined_meta.loc[m_tr, 'target_SI'],
                cmap=cmap,
                s=28, alpha=0.80, marker='o', label='Train', zorder=2
            )
        
            # Test = stars (large, outlined)
            ax.scatter(
                combined_meta.loc[m_te, 'PC1'],
                combined_meta.loc[m_te, 'PC2'],
                c=combined_meta.loc[m_te, 'target_SI'],
                cmap=cmap,
                s=160, alpha=0.95, marker='*',
                edgecolors='k', linewidths=0.5,
                label='Test', zorder=3
            )
        
            # Shared colorbar
            vmin = combined_meta['target_SI'].min()
            vmax = combined_meta['target_SI'].max()
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('DM1 Severity (Splicing Index)', fontsize=12, fontweight='bold')
        
            ax.set_xlabel(f"Principal Component 1 ({var1:.2f}%)", fontsize=14, fontweight='bold')
            ax.set_ylabel(f"Principal Component 2 ({var2:.2f}%)", fontsize=14, fontweight='bold')
            ax.set_title(f"Train → Test PCA by SI — {name_part}", fontsize=16, fontweight='bold')
            ax.legend(title='Set')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pca_train_test_overlay_by_SI_{name_part}.png"), dpi=300)
            plt.close(fig)
        
        else:
            # Fallback: color by set (no SI)
            fig, ax = plt.subplots(figsize=(12, 8))
        
            ax.scatter(
                combined_meta.loc[m_tr, 'PC1'],
                combined_meta.loc[m_tr, 'PC2'],
                s=28, alpha=0.85, marker='o',
                color='tab:blue', label='Train', zorder=2
            )
        
            ax.scatter(
                combined_meta.loc[m_te, 'PC1'],
                combined_meta.loc[m_te, 'PC2'],
                s=160, alpha=0.95, marker='*',
                color='tab:orange', edgecolors='k', linewidths=0.5,
                label='Test', zorder=3
            )
        
            ax.set_xlabel(f"Principal Component 1 ({var1:.2f}%)", fontsize=14, fontweight='bold')
            ax.set_ylabel(f"Principal Component 2 ({var2:.2f}%)", fontsize=14, fontweight='bold')
            ax.set_title(f"Train → Test PCA overlay — {name_part}", fontsize=16, fontweight='bold')
            ax.legend(title='Set')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pca_train_test_overlay_{name_part}.png"), dpi=300)
            plt.close(fig)
        


    print(f"[PCA] Finished. Figures saved to: {output_dir}")

def main():
    # Determine which sample types to load based on config
    include_types = ("DM1",) + tuple(CONTROL_LABELS) if USE_CONTROLS else ("DM1",)
    print(f"[CLASS] USE_CONTROLS={USE_CONTROLS} | CONTROL_LABELS={tuple(CONTROL_LABELS)}")
    print(f"[CLASS] include_types={include_types}")

    # ───────────────────────── Classification on COMBINED set ──────────────────
    print("\n[CLASS] Loading COMBINED dataset for classification…")
    _, _, X_combined, combined_filenames, meta_combined = load_data(
        data_dir=classification_data_dir_combined,
        metadata_path=classification_meta_path_combined,
        include_types=include_types,
        return_filenames=True,
        strict=False,
        report_samples=5,
    )

    if 'Type' not in meta_combined.columns:
        raise KeyError("Expected a 'Type' column in combined metadata; "
                       f"available columns: {list(meta_combined.columns)}")

    type_series = meta_combined['Type'].astype(str)
    n_dm1   = (type_series == 'DM1').sum()
    n_ctrl  = type_series.isin(CONTROL_LABELS).sum()
    n_total = len(type_series)
    print(f"[LOAD][COMBINED] Total rows: {n_total} | DM1: {n_dm1} | Controls (aliases={tuple(CONTROL_LABELS)}): {n_ctrl}")

    if n_dm1 == 0 or (USE_CONTROLS and n_ctrl == 0):
        print("[CLASS] Not enough classes in the COMBINED set for classification.")
        return

    clf = Classifier(all_spectra=X_combined, y_label_df=meta_combined)

    # PLS-DA (LOSOCV) on COMBINED
    print("\n[CLASS] Running PLS-DA (LOSOCV) on COMBINED…")
    roc_auc_plsda, y_test_plsda, y_pred_proba_plsda = clf.pls_da_losocv(
        preprocess_methods=plsda_preprocess,
        n_components=plsda_n_components,
        save_plots=True,
        output_dir=cls_output_dir
    )
    print(f"[CLASS][PLS-DA][COMBINED] ROC AUC (LOSOCV): {roc_auc_plsda:.4f}")

    # SVM (LOSOCV) on COMBINED
    print("\n[CLASS] Running SVM (LOSOCV) on COMBINED…")
    roc_auc_svm, y_test_svm, y_pred_proba_svm = clf.svm_losocv(
        preprocess_methods=svm_preprocess,
        C=svm_C,
        kernel=svm_kernel,
        save_plots=True,
        output_dir=cls_output_dir
    )
    print(f"[CLASS][SVM][COMBINED] ROC AUC (LOSOCV): {roc_auc_svm:.4f}")

    # Save raw LOSOCV outputs (tagged as combined)
    out_pls = os.path.join(cls_output_dir, "plsda_losocv_outputs_COMBINED.npz")
    np.savez_compressed(out_pls,
        y_test=np.array(y_test_plsda),
        y_pred_proba=np.array(y_pred_proba_plsda),
        roc_auc=np.array([roc_auc_plsda])
    )
    print(f"[CLASS][PLS-DA] Saved raw outputs → {out_pls}")

    out_svm = os.path.join(cls_output_dir, "svm_losocv_outputs_COMBINED.npz")
    np.savez_compressed(out_svm,
        y_test=np.array(y_test_svm),
        y_pred_proba=np.array(y_pred_proba_svm),
        roc_auc=np.array([roc_auc_svm])
    )
    print(f"[CLASS][SVM] Saved raw outputs → {out_svm}")

    # ──────────────────────────── PCA on separate sets ─────────────────────────
    print("\n[PCA] Running PCA analyses (train/test overlays) on separate sets…")
    run_pca_suite(
        train_data_dir=train_data_dir,
        train_meta_path=train_meta_path,
        test_data_dir=test_data_dir,
        test_meta_path=test_meta_path,
        include_types=include_types,
        output_dir=pca_output_dir
    )

    print("\n[CLASS+PCA] Done.")
    print("   • Classification figures/outputs:", cls_output_dir)
    print("   • PCA figures:", pca_output_dir)

if __name__ == "__main__":
    main()
