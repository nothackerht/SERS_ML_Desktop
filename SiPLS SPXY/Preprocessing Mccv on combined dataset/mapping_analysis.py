# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:46:20 2025

@author: spect
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from modules.data_loader import load_data, load_metadata, load_txt_file
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict
import re
from pathlib import Path
from modules.preprocessing import Preprocessing

def parse_mapping_files(data_dir):
    mapping_structure = defaultdict(lambda: defaultdict(list))
    for f in Path(data_dir).glob("*.txt"):
        match = re.match(r"(DM1_\d+|AdCo_\d+)_Map(\d)_", f.stem)
        if match:
            sample_id, map_num = match.groups()
            mapping_structure[sample_id][f"Map{map_num}"].append(str(f))
    return mapping_structure

def aggregate_mapping_spectra(data_dir, mapping_structure, which_maps, wavenumbers=None):
    spectra_per_file = 3
    collected_spectra = []
    mapping_colors = []

   

    for sample_id, maps in mapping_structure.items():
        for m in which_maps:
            for file_path in sorted(maps.get(m, [])):
                try:
                    spectra = load_txt_file(file_path)  # This returns shape (1731, 3)



                    if spectra is not None:
                        collected_spectra.append(spectra)
                        n_spectra = spectra.shape[1]  # should be 3
                        mapping_colors.extend([m] * n_spectra)


                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    if not collected_spectra:
        raise ValueError("No spectra found for the given mappings.")

    spectra_array = np.hstack(collected_spectra)
    print(f"Spectra shape: {spectra_array.shape}, Labels: {len(mapping_colors)}")

    return spectra_array, mapping_colors

def plot_all_mappings(wavenumbers, spectra, mapping_colors, output_path):
    color_map = {"Map1": "blue", "Map2": "green", "Map3": "red"}
    mask1 = wavenumbers <= 1799.44
    mask2 = wavenumbers >= 2400.51

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={'wspace': 0.05})

    for i in range(spectra.shape[1]):
        color = color_map.get(mapping_colors[i], "gray")
        ax1.plot(wavenumbers[mask1], spectra[mask1, i], color=color, alpha=0.3)
        ax2.plot(wavenumbers[mask2], spectra[mask2, i], color=color, alpha=0.3)

    ax1.set_xlim(wavenumbers[mask1][0], wavenumbers[mask1][-1])
    ax2.set_xlim(wavenumbers[mask2][0], wavenumbers[mask2][-1])

    ax1.set_xlabel("Wavenumber (cm⁻¹)")
    ax2.set_xlabel("Wavenumber (cm⁻¹)")
    ax1.set_ylabel("Intensity")
    ax1.set_title("Spectra Left Segment")
    ax2.set_title("Spectra Right Segment")
    # Add legend showing mapping colors
    for label, color in color_map.items():
        ax2.plot([], [], color=color, label=label)
    ax2.legend(loc='upper right', title='Mapping')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def run_mccv_for_mapping(X, y, preprocess_methods=None, n_iter=200, test_size=0.2, random_state=42):

    preds = np.full((len(y), n_iter), np.nan)
    for i in range(n_iter):
        try:
            train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=test_size, random_state=random_state + i)

            # Preprocessing
            if preprocess_methods:
                train_data = X[train_idx].T
                test_data = X[test_idx].T
                prep = Preprocessing(train_data)
                train_data = prep.preprocess(preprocess_methods)

                if 'EMSC' in preprocess_methods:
                    # Use training reference for EMSC
                    reference = np.mean(train_data, axis=1)
                    prep_test = Preprocessing(test_data)
                    test_data = prep_test.emsc(test_data, reference=reference)
                else:
                    prep_test = Preprocessing(test_data)
                    test_data = prep_test.preprocess(preprocess_methods)

                X_train = train_data.T
                X_test = test_data.T
            else:
                X_train = X[train_idx]
                X_test = X[test_idx]

            pls = PLSRegression(n_components=5)
            pls.fit(X_train, y[train_idx])
            pred = pls.predict(X_test).flatten()
            preds[test_idx, i] = pred

        except Exception as e:
            print(f"[MCCV Error] Iter {i}: {e}")
            continue

    mean_preds = np.nanmean(preds, axis=1)
    r2 = r2_score(y, mean_preds)
    rmse = np.sqrt(mean_squared_error(y, mean_preds))
    return r2, rmse, mean_preds, preds


def plot_separated_mappings(wavenumbers, spectra, mapping_colors, output_dir):
    color_map = {"Map1": "blue", "Map2": "green", "Map3": "red"}
    unique_maps = ["Map1", "Map2", "Map3"]
    mask1 = wavenumbers <= 1799.44
    mask2 = wavenumbers >= 2400.51

    for m in unique_maps:
        indices = [i for i, label in enumerate(mapping_colors) if label == m]
        selected_spectra = spectra[:, indices]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={'wspace': 0.05})

        for i in range(selected_spectra.shape[1]):
            ax1.plot(wavenumbers[mask1], selected_spectra[mask1, i], color=color_map[m], alpha=0.3)
            ax2.plot(wavenumbers[mask2], selected_spectra[mask2, i], color=color_map[m], alpha=0.3)

        ax1.set_xlim(wavenumbers[mask1][0], wavenumbers[mask1][-1])
        ax2.set_xlim(wavenumbers[mask2][0], wavenumbers[mask2][-1])

        ax1.set_xlabel("Wavenumber (cm⁻¹)")
        ax2.set_xlabel("Wavenumber (cm⁻¹)")
        ax1.set_ylabel("Intensity")
        ax1.set_title(f"{m} Spectra (Left Segment)")
        ax2.set_title(f"{m} Spectra (Right Segment)")
        # Add legend for this specific mapping
        ax2.plot([], [], color=color_map[m], label=m)
        ax2.legend(loc='upper right', title='Mapping')


        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"Overlay_{m}_Only.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
def plot_individual_mapping_overlay(wavenumbers, mapping_structure, data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    color_map = {"Map1": "blue", "Map2": "green", "Map3": "red"}
    masks = {
        "left": wavenumbers <= 1799.44,
        "right": wavenumbers >= 2400.51
    }

    for map_label in ["Map1", "Map2", "Map3"]:
        collected = []

        for sample_id, maps in mapping_structure.items():
            if map_label in maps:
                for file_path in maps[map_label]:
                    spectra = load_txt_file(file_path)  # shape (1731, 3)
                    if spectra is not None:
                        collected.append(spectra)

        if not collected:
            continue

        full = np.hstack(collected)  # shape (1731, N)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={'wspace': 0.05})

        for i in range(full.shape[1]):
            ax1.plot(wavenumbers[masks["left"]], full[masks["left"], i], color=color_map[map_label], alpha=0.3)
            ax2.plot(wavenumbers[masks["right"]], full[masks["right"], i], color=color_map[map_label], alpha=0.3)

        ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=14)
        ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=14)
        ax1.set_ylabel("Intensity", fontsize=14)
        ax1.set_xlim(wavenumbers[masks["left"]][0], wavenumbers[masks["left"]][-1])
        ax2.set_xlim(wavenumbers[masks["right"]][0], wavenumbers[masks["right"]][-1])
        fig.suptitle(f"{map_label} Spectra Overlay", fontsize=16, fontweight='bold')
        # Add legend showing current mapping
        ax2.plot([], [], color=color_map[map_label], label=map_label)
        ax2.legend(loc='upper right', title='Mapping')


        output_path = os.path.join(output_dir, f"Overlay_{map_label}_SplitStyle.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
def analyze_std_outliers(std_preds, mean_preds, label, preprocess_label, output_dir):
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    std_mean = np.mean(std_preds)
    std_std = np.std(std_preds)
    threshold = std_mean + 2 * std_std

    outlier_indices = np.where(std_preds > threshold)[0]

    print(f"[STD OUTLIERS] {label} | {preprocess_label} — Outliers: {len(outlier_indices)} above threshold {threshold:.3f}")

    # Save outlier info
    outlier_df = pd.DataFrame({
        'Index': outlier_indices,
        'MeanPrediction': mean_preds[outlier_indices],
        'STD': std_preds[outlier_indices]
    })
    filename = f"STD_Outliers_{label.replace('+', '_')}_{preprocess_label}.xlsx"
    outlier_df.to_excel(os.path.join(output_dir, filename), index=False)

    # Plot with threshold and highlight outliers
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mean_preds, std_preds, alpha=0.5, c='gray', edgecolor='k', label='Inliers')
    ax.scatter(mean_preds[outlier_indices], std_preds[outlier_indices], color='red', edgecolor='k', label='Outliers')
    ax.axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
    ax.set_xlabel("Mean Prediction")
    ax.set_ylabel("Prediction STD")
    ax.set_title(f"{label} — STD Outlier Detection\n{preprocess_label}")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"STD_Outliers_{label.replace('+', '_')}_{preprocess_label}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def run_mapping_comparison(data_dir, metadata_path, output_dir, target='target_SI'):
    os.makedirs(output_dir, exist_ok=True)
    wavenumbers, _, _ = load_data(data_dir)
    y_df = load_metadata(metadata_path)
    if "Sample_ID" in y_df.columns:
        y_df.set_index("Sample_ID", inplace=True)

    # 👇 Fix NaN SI values based on Type
    control_mask = y_df['Type'] == 'Control'
    si_is_nan   = y_df['target_SI'].isna()

    # Assign Control NaNs → 0
    y_df.loc[control_mask & si_is_nan, 'target_SI'] = 0

    # Drop non-Control NaNs
    y_df = y_df[~((~control_mask) & si_is_nan)].copy()

    print("Metadata columns:", y_df.columns)
    print("Metadata index example:", list(y_df.index[:5]))

    mapping_structure = parse_mapping_files(data_dir)

    print("Sample IDs from filenames:", list(mapping_structure.keys())[:5])
    print("Metadata index sample IDs:", list(y_df.index[:5]))
    print("Example unmatched sample_id:", [sid for sid in mapping_structure.keys() if sid not in y_df.index][:5])

    # ── Paste the four one‐time overlay calls right here ──
    plot_individual_mapping_overlay(wavenumbers, mapping_structure, data_dir, output_dir)

    full_spectra, color_labels = aggregate_mapping_spectra(
        data_dir,
        mapping_structure,
        ["Map1", "Map2", "Map3"],
        wavenumbers=wavenumbers
    )

    plot_all_mappings(
        wavenumbers,
        full_spectra,
        color_labels,
        os.path.join(output_dir, "Overlay_All_Mappings.png")
    )

    plot_separated_mappings(wavenumbers, full_spectra, color_labels, output_dir)
    # ────────────────────────────────────────────────────

    results = []

    map_sets = {
        "Map1": ["Map1"],
        "Map2": ["Map2"],
        "Map3": ["Map3"],
        "Map1+2": ["Map1", "Map2"],
        "Map1+3": ["Map1", "Map3"],
        "Map2+3": ["Map2", "Map3"],
        "Map1+2+3": ["Map1", "Map2", "Map3"]
    }
    preprocessing_options = [
        [],  # No preprocessing
        ['EMSC'],
        ['Normalization'],
        ['SNV'],
        ['Second Derivative'],
        ['EMSC', 'Normalization'],
        ['EMSC', 'SNV'],
        ['EMSC', 'Second Derivative']
    ]

    for preprocess_methods in preprocessing_options:
        preprocess_label = '+'.join(preprocess_methods) if preprocess_methods else 'None'

        for label, maps in map_sets.items():
            print(f"\n🧪 Evaluating {label} with preprocessing: {preprocess_label}")
            X, colors = aggregate_mapping_spectra(data_dir, mapping_structure, maps, wavenumbers=wavenumbers)
    
            sample_ids = []
            for sample_id, maps_all in mapping_structure.items():
                if all(m in maps_all for m in maps):
                    sample_ids.append(sample_id)
    
            sample_ids = [sid for sid in sample_ids if sid in y_df.index]
            y_clean = y_df.loc[sample_ids, target]
            print(f"[DEBUG] Skipped samples in {label}:")
    
            y_parts = []
            for sid in sample_ids:
                count = 0
                if sid not in y_df.index:
                    print(f" - {sid} missing in metadata")
                    continue
                elif pd.isna(y_clean.get(sid)):
                    print(f" - {sid} has NaN target value")
                    continue
                else:
                    for m in maps:
                        count += len(mapping_structure[sid][m]) * 3
                    if count == 0:
                        print(f" - {sid} has no spectra for maps {maps}")
                    else:
                        y_parts.append(np.repeat(y_clean[sid], count))
    
            if not y_parts:
                print(f"Skipping {label} due to no spectra matched with y.")
                continue
    
            y = np.concatenate(y_parts)
            if X.shape[1] > len(y):
                print(f"[WARN] More spectra ({X.shape[1]}) than labels ({len(y)}). Trimming spectra array.")
                X = X[:, :len(y)]
            elif X.shape[1] < len(y):
                print(f"[ERROR] More labels than spectra. Skipping {label}.")
                continue
    
            X = X[:, :len(y)]  # trim spectra if too many
            X = X.T
    
            # 🧪 Run MCCV with preprocessing
            r2, rmse, y_pred, preds = run_mccv_for_mapping(X, y, preprocess_methods=preprocess_methods)
    
            # Plot and save Mean vs STD
            mean_preds = np.nanmean(preds, axis=1)
            std_preds = np.nanstd(preds, axis=1)
    
            fig, ax = plt.subplots(figsize=(6, 6))
            sc = ax.scatter(mean_preds, std_preds, alpha=0.5, c='black', edgecolor='k')
            ax.set_xlabel("Mean Prediction", fontsize=12)
            ax.set_ylabel("Prediction STD", fontsize=12)
            ax.set_title(f"{label} — Mean vs STD\n{preprocess_label}", fontsize=13, fontweight='bold')
            plt.tight_layout()
            std_plot_path = os.path.join(output_dir, f"Mean_STD_{label.replace('+', '_')}_{preprocess_label}.png")
            plt.savefig(std_plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            analyze_std_outliers(std_preds, mean_preds, label, preprocess_label, output_dir)

            print(f"{label} ({preprocess_label}) — R²: {r2:.4f}, RMSE: {rmse:.4f}")
            print(f"✔️ Finished — saving plots.")
    
            results.append({'Mapping': label, 'Preprocessing': preprocess_label, 'R2': r2, 'RMSE': rmse})
    
            # Parity plot
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y, y_pred, alpha=0.5, edgecolor='k')
            ax.plot([min(y), max(y)], [min(y), max(y)], 'r--', linewidth=1.5)
            ax.set_xlabel("True Values", fontsize=12, fontweight='bold')
            ax.set_ylabel("Predicted Values", fontsize=12, fontweight='bold')
            ax.set_title(f"{label} Parity Plot\n{preprocess_label}\nR² = {r2:.3f}, RMSE = {rmse:.2f}", fontsize=13, fontweight='bold')
            plt.tight_layout()
            parity_plot_path = os.path.join(output_dir, f"ParityPlot_{label.replace('+', '_')}_{preprocess_label}.png")
            plt.savefig(parity_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
            # Residual histogram
            residuals = y_pred - y
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(residuals, bins=30, color='gray', edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--')
            ax.set_title(f"{label} Residuals Histogram\n{preprocess_label}", fontsize=13, fontweight='bold')
            ax.set_xlabel("Prediction Error (y_pred - y_true)", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            plt.tight_layout()
            hist_path = os.path.join(output_dir, f"ResidualsHist_{label.replace('+', '_')}_{preprocess_label}.png")
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    

    
        compute_average_baseline_per_mapping(data_dir, mapping_structure, wavenumbers)
        # === Baseline Boxplot and ANOVA Analysis ===
        import seaborn as sns
        import scipy.stats as stats
        
        print("Generating baseline distribution analysis across mappings...")
        
        baseline_df = pd.DataFrame(columns=['SampleID', 'Map', 'Baseline'])
        
        for sample_id, maps in mapping_structure.items():
            for map_label in ['Map1', 'Map2', 'Map3']:
                if map_label in maps:
                    for file_path in maps[map_label]:
                        spectra = load_txt_file(file_path)  # shape: (wavenumbers, 3)
                        for i in range(spectra.shape[1]):
                            baseline = np.mean(spectra[:200, i])  # First 200 points
                            baseline_df.loc[len(baseline_df)] = [sample_id, map_label, baseline]
        
        # Plot and save baseline distribution
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=baseline_df, x='Map', y='Baseline', palette={"Map1":"blue", "Map2":"green", "Map3":"red"})
        plt.title("Baseline Distribution per Mapping", fontsize=14, fontweight='bold')
        plt.ylabel("Mean Baseline (First 200 points)", fontsize=12)
        plt.xlabel("Mapping", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        # Save and show
        baseline_plot_path = os.path.join(output_dir, "Baseline_Distribution_Boxplot.png")
        plt.savefig(baseline_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical test
        map1_vals = baseline_df[baseline_df['Map'] == 'Map1']['Baseline']
        map2_vals = baseline_df[baseline_df['Map'] == 'Map2']['Baseline']
        map3_vals = baseline_df[baseline_df['Map'] == 'Map3']['Baseline']
        
        f_stat, p_val = stats.f_oneway(map1_vals, map2_vals, map3_vals)
        print(f"\n📊 ANOVA result for baseline shift: F = {f_stat:.4f}, p = {p_val:.4e}")
        
        # Save to Excel
        baseline_df.to_excel(os.path.join(output_dir, "Baseline_Distribution_Data.xlsx"), index=False)
    

    
        print("✓ Mapping comparison complete.")
        print("Generating per-sample mapping overlay plots...")
        per_sample_plot_dir = os.path.join(output_dir, "Per_Sample_Overlay_Plots")
        os.makedirs(per_sample_plot_dir, exist_ok=True)
    
        color_map = {"Map1": "blue", "Map2": "green", "Map3": "red"}
        masks = {
            "left": wavenumbers <= 1799.44,
            "right": wavenumbers >= 2400.51
        }
    
        for sample_id, maps in mapping_structure.items():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={'wspace': 0.05})
    
            plotted = False  # Track if anything was plotted
    
            for map_label in ["Map1", "Map2", "Map3"]:
                if map_label in maps:
                    for file_path in maps[map_label]:
                        spectra = load_txt_file(file_path)  # shape (1731, 3)
                        if spectra is not None:
                            for i in range(spectra.shape[1]):
                                ax1.plot(wavenumbers[masks["left"]], spectra[masks["left"], i], color=color_map[map_label], alpha=0.5)
                                ax2.plot(wavenumbers[masks["right"]], spectra[masks["right"], i], color=color_map[map_label], alpha=0.5)
                            plotted = True
    
            if plotted:
                ax1.set_xlabel("Wavenumber (cm⁻¹)")
                ax2.set_xlabel("Wavenumber (cm⁻¹)")
                ax1.set_ylabel("Intensity")
                ax1.set_xlim(wavenumbers[masks["left"]][0], wavenumbers[masks["left"]][-1])
                ax2.set_xlim(wavenumbers[masks["right"]][0], wavenumbers[masks["right"]][-1])
                fig.suptitle(f"{sample_id} — All Map Spectra Overlay", fontsize=14, fontweight='bold')
                # Add legend showing all mappings
                for label, color in color_map.items():
                    ax2.plot([], [], color=color, label=label)
                ax2.legend(loc='upper right', title='Mapping')
                
    
                out_path = os.path.join(per_sample_plot_dir, f"{sample_id}_Overlay.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
    
                # Show in plot window
                plt.show()

                plt.close()


    # === POST-HOC PAIRWISE TESTS (after ANOVA) ===
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from scipy.stats import ttest_ind
    import statsmodels.api as sm
    
    print("\nRunning post-hoc pairwise comparisons:")
    
    # Tukey's HSD test
    tukey = pairwise_tukeyhsd(endog=baseline_df['Baseline'],
                              groups=baseline_df['Map'],
                              alpha=0.05)
    
    print(tukey)
    # Save Tukey HSD results to a text file
    with open(os.path.join(output_dir, "TukeyHSD_Baseline_Results.txt"), "w") as f:
        f.write(str(tukey.summary()))

    # Pairwise t-tests with Bonferroni correction
    maps = ['Map1', 'Map2', 'Map3']
    print("\nPairwise t-tests with Bonferroni correction:")
    # Save t-test results
    ttest_path = os.path.join(output_dir, "TTest_Baseline_Results.txt")
    with open(ttest_path, "w") as f:
        f.write("Pairwise t-tests with Bonferroni correction:\n\n")
        for i in range(len(maps)):
            for j in range(i+1, len(maps)):
                m1 = baseline_df[baseline_df['Map'] == maps[i]]['Baseline']
                m2 = baseline_df[baseline_df['Map'] == maps[j]]['Baseline']
                stat, pval = ttest_ind(m1, m2)
                corrected_p = pval * 3
                f.write(f"{maps[i]} vs {maps[j]}: t = {stat:.4f}, p = {pval:.4e}, corrected p = {corrected_p:.4e}\n")

    for i in range(len(maps)):
        for j in range(i+1, len(maps)):
            m1 = baseline_df[baseline_df['Map'] == maps[i]]['Baseline']
            m2 = baseline_df[baseline_df['Map'] == maps[j]]['Baseline']
            stat, pval = ttest_ind(m1, m2)
            corrected_p = pval * 3  # Bonferroni correction
            print(f"{maps[i]} vs {maps[j]}: t = {stat:.4f}, p = {pval:.4e}, corrected p = {corrected_p:.4e}")

def compute_average_baseline_per_mapping(data_dir, mapping_structure, wavenumbers):
    color_map = {"Map1": "blue", "Map2": "green", "Map3": "red"}
    baseline_data = {"Map1": [], "Map2": [], "Map3": []}

    for sample_id, maps in mapping_structure.items():
        for map_label in ["Map1", "Map2", "Map3"]:
            if map_label in maps:
                for file_path in maps[map_label]:
                    spectra = load_txt_file(file_path)  # shape (1731, 3)
                    if spectra is not None:
                        for i in range(spectra.shape[1]):
                            mean_baseline = np.mean(spectra[:, i])
                            baseline_data[map_label].append(mean_baseline)

    results = []
    
    for map_label in ["Map1", "Map2", "Map3"]:
        values = baseline_data[map_label]
        avg = np.mean(values) if values else np.nan
        std = np.std(values) if values else np.nan
        results.append((map_label, avg, std))

    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

    print("\n🔬 Average Baseline Ranking Across Mappings:")
    for label, avg, std in results_sorted:
        print(f"{label}: Mean Baseline = {avg:.2f}, Std = {std:.2f}")

    return results_sorted
