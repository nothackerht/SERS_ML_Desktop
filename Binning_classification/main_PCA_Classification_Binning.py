# -*- coding: utf-8 -*-
"""
main_binning_and_pca.py

Runs:
  1) PCA analyses (train-only & train→test overlays with optional SI coloring)
     on separate TRAIN and TEST sets.
  2) SI binning classification (low/med/high) using grouped/replicate method
     with GroupKFold CV on the COMBINED set.

Outputs:
  • PCA figures   → pca_output_dir
  • Binning results (CSV + mean confusion matrices)
                   → BIN_OUT_DIR\SI_bin_cls_grouped
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix

# --- Project imports (your modules) ---
from modules.data_loader import load_data
from modules.unsupervised_clustering import UnsupervisedClustering
from modules.preprocessing import Preprocessing

# ============================ USER PATHS ============================

# COMBINED data (for binning classification)
classification_data_dir_combined  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\data_combined"
classification_meta_path_combined = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata_combined_in_order.csv"

# Separate TRAIN data (for PCA)
train_data_dir  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
train_meta_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"

# Separate TEST data (for PCA)
test_data_dir  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data_test_updated"
test_meta_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata_test_updated.csv"

# Outputs
pca_output_dir = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\PCA_Results"
BIN_OUT_DIR    = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Binning_classification\Results"
os.makedirs(pca_output_dir, exist_ok=True)
os.makedirs(BIN_OUT_DIR, exist_ok=True)

# ============================ CONSTANTS ============================
INCLUDE_TYPES = ("DM1",)   # DM1-only; add Control labels if you want them included
REPS = 9                   # 9 replicate spectra per sample

# ============================ HELPERS ==============================

def si_to_bins(si_series: pd.Series) -> pd.Series:
    """Bins: [0,0.33), [0.33,0.66), [0.66,1.0]."""
    return pd.cut(
        pd.to_numeric(si_series, errors="coerce"),
        bins=[-1e-9, 0.33, 0.66, 1.0 + 1e-9],
        labels=["low", "med", "high"],
        right=False,
        include_lowest=True,
    )

def _groups_for_spectra(n_samples: int, reps: int = REPS) -> np.ndarray:
    return np.repeat(np.arange(n_samples, dtype=int), reps)

# ============================ PCA SUITE ============================

def run_pca_suite(
    train_data_dir: str,
    train_meta_path: str,
    test_data_dir: str,
    test_meta_path: str,
    include_types: tuple,
    output_dir: str,
):
    """
    PCA train-only plots + train→test overlay. Uses grouped (replicate-level) spectra internally,
    then converts to samples×features for sklearn PCA.
    """
    wavs_train, _, spectra_train, _fn_tr, meta_train = load_data(
        data_dir=train_data_dir,
        metadata_path=train_meta_path,
        include_types=include_types,
        return_filenames=True,
        strict=False,
        report_samples=5,
    )
    wavs_test, _, spectra_test, _fn_te, meta_test = load_data(
        data_dir=test_data_dir,
        metadata_path=test_meta_path,
        include_types=include_types,
        return_filenames=True,
        strict=False,
        report_samples=5,
    )

    # Preprocessing combos
    preprocessing_combos = [
        [], ['Normalization'], ['SNV'], ['EMSC'], ['Second Derivative'],
        ['Normalization', 'SNV'], ['EMSC', 'Second Derivative'],
    ]

    # Train-only PCA plots via your clustering helper (replicate-level input)
    clustering_train = UnsupervisedClustering(meta_train, spectra_train, output_dir)

    for methods in preprocessing_combos:
        name_part = '__'.join(methods) if methods else 'no_preprocessing'

        # 1) Train-only PCA (using your helper)
        clustering_train.perform_pca_and_plot(
            x_array=spectra_train,
            plot_name=f"pca_train_only_pc1_vs_pc2_{name_part}",
            pc_axes=(1, 2),
            save_plot=True,
            preprocessing=methods
        )

        # 1b) Train-only PCA with SI coloring (if available)
        if 'target_SI' in meta_train.columns:
            clustering_train.perform_pca_and_plot_with_severity(
                x_array=spectra_train,
                plot_name=f"pca_train_only_pc1_vs_pc2_severity_{name_part}",
                pc_axes=(1, 2),
                save_plot=True,
                preprocessing=methods
            )

        # 2) Train→Test overlay with sklearn PCA (fit on TRAIN, transform TEST)
        if methods:
            pre = Preprocessing()
            pre.fit(spectra_train, methods=methods)
            X_train_prep = pre.transform(spectra_train, methods=methods)
            X_test_prep  = pre.transform(spectra_test,  methods=methods)
        else:
            X_train_prep, X_test_prep = spectra_train, spectra_test

        # sklearn expects samples×features
        Xtr = X_train_prep.T
        Xte = X_test_prep.T

        pca = PCA(n_components=3)
        train_scores = pca.fit_transform(Xtr)
        test_scores  = pca.transform(Xte)

        # Build metadata repeated 9× (one row per spectrum) for coloring & joining
        meta_train_rep = pd.DataFrame(np.repeat(meta_train.values, REPS, axis=0), columns=meta_train.columns)
        meta_test_rep  = pd.DataFrame(np.repeat(meta_test.values,  REPS, axis=0), columns=meta_test.columns)
        meta_train_rep['Set'] = 'Train'
        meta_test_rep['Set']  = 'Test'

        combined_meta = pd.concat([meta_train_rep, meta_test_rep], ignore_index=True)
        combined_meta['PC1'] = np.concatenate([train_scores[:, 0], test_scores[:, 0]])
        combined_meta['PC2'] = np.concatenate([train_scores[:, 1], test_scores[:, 1]])

        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100

        m_tr = combined_meta['Set'] == 'Train'
        m_te = combined_meta['Set'] == 'Test'

        if 'target_SI' in combined_meta.columns:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            fig, ax = plt.subplots(figsize=(12, 8))
            cmap = 'viridis'

            ax.scatter(
                combined_meta.loc[m_tr, 'PC1'],
                combined_meta.loc[m_tr, 'PC2'],
                c=combined_meta.loc[m_tr, 'target_SI'],
                cmap=cmap, s=28, alpha=0.80, marker='o', label='Train', zorder=2
            )
            ax.scatter(
                combined_meta.loc[m_te, 'PC1'],
                combined_meta.loc[m_te, 'PC2'],
                c=combined_meta.loc[m_te, 'target_SI'],
                cmap=cmap, s=54, alpha=0.95, marker='s',
                edgecolors='k', linewidths=0.5, label='Test', zorder=3
            )

            vmin = combined_meta['target_SI'].min()
            vmax = combined_meta['target_SI'].max()
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('DM1 Severity (Splicing Index)', fontsize=12, fontweight='bold')
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(
                combined_meta.loc[m_tr, 'PC1'], combined_meta.loc[m_tr, 'PC2'],
                s=28, alpha=0.85, marker='o', color='tab:blue', label='Train', zorder=2
            )
            ax.scatter(
                combined_meta.loc[m_te, 'PC1'], combined_meta.loc[m_te, 'PC2'],
                s=56, alpha=0.95, marker='s', color='tab:orange', edgecolors='k', linewidths=0.5,
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

# ==================== BINNING CLASSIFICATION (GROUPED) ====================

def run_si_bin_classification_grouped(
    data_dir: str, meta_path: str, out_dir: str,
    include_types=("DM1",), n_splits=5, random_state=42
):
    """
    Bins SI into low/med/high, trains at the replicate level (9× per sample),
    uses GroupKFold (no replicate leakage), and scores at the sample level
    via averaged probabilities across the 9 replicates.
    """
    wavs, avg, all_spec, _names, meta = load_data(
        data_dir=data_dir,
        metadata_path=meta_path,
        include_types=include_types,
        return_filenames=True,
        strict=False,
        report_samples=3,
    )

    # Replicate-level features
    X_all = all_spec.T.astype(np.float32)  # (9N, 1731)
    N = meta.shape[0]

    # Sample-level SI → bins
    si = pd.to_numeric(meta["target_SI"], errors="coerce")
    keep_samples = si.notna().to_numpy()
    if keep_samples.sum() < 2:
        raise ValueError("Not enough samples with valid SI to classify.")

    # Filter to kept samples at replicate level
    keep_spec = np.repeat(keep_samples, REPS)
    X = X_all[keep_spec]

    # Labels & groups
    y_sample = si_to_bins(si[keep_samples]).astype(str).to_numpy()     # (N_keep,)
    y_spec   = np.repeat(y_sample, REPS)                               # (9*N_keep,)
    groups   = _groups_for_spectra(int(keep_samples.sum()), REPS)      # (9*N_keep,)

    # Class sanity and folds
    classes, counts = np.unique(y_sample, return_counts=True)
    print("[CLS] sample-level class counts:", dict(zip(classes, counts)))
    assert len(classes) >= 2, "Need at least 2 classes after binning."
    n_folds = max(2, int(min(n_splits, counts.min())))
    # --- ADD: class count bar chart (once) ---
    fig, ax = plt.subplots(figsize=(5, 3))
    label_order = ["low", "med", "high"]  # ensure consistent order
    counts_map = dict(zip(classes, counts))
    ax.bar(label_order, [counts_map.get(c, 0) for c in label_order])
    ax.set_title("Class counts (samples)")
    ax.set_ylabel("N")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "class_counts.png"), dpi=300)
    plt.close(fig)
    # -----------------------------------------

    # Models (probability-enabled where possible)
    candidates = {
        "logreg": make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=2000, multi_class="multinomial",
                               class_weight="balanced", C=1.0, random_state=random_state)
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            SVC(kernel="rbf", C=2.0, gamma="scale",
                class_weight="balanced", probability=True, random_state=random_state)
        ),
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=None, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1
        ),
    }

    gkf = GroupKFold(n_splits=n_folds)
    label_order = ["low", "med", "high"]
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for name, model in candidates.items():
        f1s, accs, baccs = [], [], []
        cm_sum = np.zeros((3, 3), dtype=float)

        for tr_idx, te_idx in gkf.split(X, y_spec, groups):
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y_spec[tr_idx], y_spec[te_idx]
            gte = groups[te_idx]

            model.fit(Xtr, ytr)

            # Predict spectrum-level probs, average per-sample in the test fold
            if hasattr(model, "predict_proba"):
                proba_te = model.predict_proba(Xte)  # (9M, 3)
                _, inv, counts_fold = np.unique(gte, return_inverse=True, return_counts=True)
                M = counts_fold.size
                prob_avg = np.zeros((M, len(label_order)), dtype=float)
                np.add.at(prob_avg, inv, proba_te)
                prob_avg /= counts_fold[:, None]
                yhat_s = np.array(label_order)[prob_avg.argmax(axis=1)]
            else:
                # Fallback: majority vote on class labels
                yhat = model.predict(Xte)
                _, inv, counts_fold = np.unique(gte, return_inverse=True, return_counts=True)
                M = counts_fold.size
                votes = np.zeros((M, len(label_order)), dtype=int)
                for k, c in enumerate(label_order):
                    votes[:, k] = np.bincount(inv[yhat == c], minlength=M)
                yhat_s = np.array(label_order)[votes.argmax(axis=1)]

            # True sample labels for this fold (one per group)
            yte_s = yte[::REPS][:len(np.unique(gte))]

            # Metrics (sample level)
            f1s.append(f1_score(yte_s, yhat_s, average="macro", labels=label_order))
            accs.append(accuracy_score(yte_s, yhat_s))
            baccs.append(balanced_accuracy_score(yte_s, yhat_s))
            cm_sum += confusion_matrix(yte_s, yhat_s, labels=label_order)

        res = {
            "model": name,
            "f1_macro_mean": float(np.mean(f1s)),
            "f1_macro_std":  float(np.std(f1s)),
            "acc_mean":       float(np.mean(accs)),
            "bacc_mean":      float(np.mean(baccs)),
            "n_folds":        int(n_folds),
        }
        results.append(res)

        # Save mean confusion matrix
        cm_mean = (cm_sum / n_folds).astype(float)
        pd.DataFrame(cm_mean, index=label_order, columns=label_order)\
          .to_csv(os.path.join(out_dir, f"{name}__cm_mean.csv"))
        # --- ADD: normalized confusion-matrix heatmap (per model) ---
        cm_safe = cm_mean.copy()
        row_sums = cm_safe.sum(axis=1, keepdims=True)
        # avoid divide-by-zero if a class is absent in a fold set
        row_sums[row_sums == 0] = 1.0
        cm_norm = cm_safe / row_sums
        
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm_norm, vmin=0, vmax=1)
        ax.set_xticks(range(len(label_order))); ax.set_xticklabels(label_order)
        ax.set_yticks(range(len(label_order))); ax.set_yticklabels(label_order)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{name} — Confusion (norm)")
        # annotate cells
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}__cm_mean.png"), dpi=300)
        plt.close(fig)
        # ------------------------------------------------------------

        print(f"[{name}] F1_macro={res['f1_macro_mean']:.3f}±{res['f1_macro_std']:.3f} "
              f"| Acc={res['acc_mean']:.3f} | BalAcc={res['bacc_mean']:.3f} | folds={n_folds}")

    df = pd.DataFrame(results).sort_values("f1_macro_mean", ascending=False)
    df.to_csv(os.path.join(out_dir, "si_bin_cls_grouped_results.csv"), index=False)
    print("\n[RESULTS]\n", df)
    print("[SAVE] →", os.path.join(out_dir, "si_bin_cls_grouped_results.csv"))

# ============================ MAIN ============================

def main():
    include_types = INCLUDE_TYPES
    print(f"[CONFIG] include_types = {include_types}")

    # --- PCA on separate TRAIN/TEST sets ---
    print("\n[PCA] Running PCA analyses (train/test overlays)…")
    run_pca_suite(
        train_data_dir=train_data_dir,
        train_meta_path=train_meta_path,
        test_data_dir=test_data_dir,
        test_meta_path=test_meta_path,
        include_types=include_types,
        output_dir=pca_output_dir
    )

    # --- SI binning classification on COMBINED set (grouped) ---
    print("\n[CLS] Running SI binning classification (grouped)…")
    run_si_bin_classification_grouped(
        data_dir=classification_data_dir_combined,
        meta_path=classification_meta_path_combined,
        out_dir=os.path.join(BIN_OUT_DIR, "SI_bin_cls_grouped"),
        include_types=include_types,
        n_splits=5,
        random_state=42,
    )

    print("\n[DONE]")
    print("  • PCA figures:", pca_output_dir)
    print("  • Binning classification results:", BIN_OUT_DIR)

if __name__ == "__main__":
    main()
