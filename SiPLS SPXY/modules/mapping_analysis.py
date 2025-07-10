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

# ─── Paths to your metadata CSVs (for the Combined run coloring) ───
train_data_dir = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
train_meta_csv = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"

test_data_dir = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data_test_updated"
test_meta_csv = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata_test_updated.csv"
def parse_mapping_files(data_dir):
    mapping_structure = defaultdict(lambda: defaultdict(list))
    for f in Path(data_dir).glob("*.txt"):
        match = re.match(r"(DM1_\d+|AdCo_\d+)_Map(\d)_", f.stem)
        if match:
            sample_id, map_num = match.groups()
            mapping_structure[sample_id][f"Map{map_num}"].append(str(f))
    return mapping_structure


def aggregate_mapping_spectra(data_dir, mapping_structure, which_maps, wavenumbers=None):
    collected_spectra = []
    mapping_colors = []

    for sample_id, maps in mapping_structure.items():
        for m in which_maps:
            for file_path in sorted(maps.get(m, [])):
                try:
                    spectra = load_txt_file(file_path)  # (1731, 3)
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

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), sharey=True, gridspec_kw={"wspace": 0.05}
    )

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
    for label, color in color_map.items():
        ax2.plot([], [], color=color, label=label)
    ax2.legend(loc="upper right", title="Mapping")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_separated_mappings(wavenumbers, spectra, mapping_colors, output_dir):
    color_map = {"Map1": "blue", "Map2": "green", "Map3": "red"}
    unique_maps = ["Map1", "Map2", "Map3"]
    mask1 = wavenumbers <= 1799.44
    mask2 = wavenumbers >= 2400.51

    for m in unique_maps:
        indices = [i for i, label in enumerate(mapping_colors) if label == m]
        selected_spectra = spectra[:, indices]

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6), sharey=True, gridspec_kw={"wspace": 0.05}
        )
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
        ax2.plot([], [], color=color_map[m], label=m)
        ax2.legend(loc="upper right", title="Mapping")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"Overlay_{m}_Only.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


def plot_individual_mapping_overlay(wavenumbers, mapping_structure, data_dir, output_base_dir):
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    os.makedirs(output_base_dir, exist_ok=True)
    same_scale_dir = os.path.join(output_base_dir, "Same_Scale_Per_sample_Spectra")
    os.makedirs(same_scale_dir, exist_ok=True)

    color_map = {"Map1": "blue", "Map2": "green", "Map3": "red"}
    masks = {"left": wavenumbers <= 1799.44, "right": wavenumbers >= 2400.51}

    for sample_id, maps in mapping_structure.items():
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6), sharey=True, gridspec_kw={"wspace": 0.05}
        )

        for map_label in ["Map1", "Map2", "Map3"]:
            if map_label in maps:
                for file_path in maps[map_label]:
                    spectra = load_txt_file(file_path)  # (1731, 3)
                    if spectra is not None:
                        for i in range(spectra.shape[1]):
                            ax1.plot(wavenumbers[masks["left"]], spectra[masks["left"], i], color=color_map[map_label], alpha=0.3)
                            ax2.plot(wavenumbers[masks["right"]], spectra[masks["right"], i], color=color_map[map_label], alpha=0.3)

        # Apply same y-axis scaling to all plots
        ax1.set_ylim(0, 20000)
        ax2.set_ylim(0, 20000)

        ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=14)
        ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=14)
        ax1.set_ylabel("Intensity", fontsize=14)
        ax1.set_xlim(wavenumbers[masks["left"]][0], wavenumbers[masks["left"]][-1])
        ax2.set_xlim(wavenumbers[masks["right"]][0], wavenumbers[masks["right"]][-1])
        fig.suptitle(f"{sample_id} — All Map Spectra Overlay", fontsize=16, fontweight="bold")

        # Custom legend
        for label, color in color_map.items():
            ax2.plot([], [], color=color, label=label)
        ax2.legend(loc="upper right", title="Mapping")

        output_path = os.path.join(same_scale_dir, f"{sample_id}_Overlay_SameScale.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {output_path}")


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

                if "EMSC" in preprocess_methods:
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


def run_mapping_comparison(data_dir, metadata_path, output_dir, target="target_SI"):
    os.makedirs(output_dir, exist_ok=True)
    wavenumbers, _, _ = load_data(data_dir)
    y_df = load_metadata(metadata_path)
    if "Sample_ID" in y_df.columns:
        y_df.set_index("Sample_ID", inplace=True)

    # Fix NaN SI for controls, drop other NaNs
    control_mask = y_df["Type"] == "Control"
    si_is_nan = y_df["target_SI"].isna()
    y_df.loc[control_mask & si_is_nan, "target_SI"] = 0
    y_df = y_df[~((~control_mask) & si_is_nan)].copy()

    mapping_structure = parse_mapping_files(data_dir)

    map_sets = {
        "Map1": ["Map1"],
        "Map2": ["Map2"],
        "Map3": ["Map3"],
        "Map1+2": ["Map1", "Map2"],
        "Map1+3": ["Map1", "Map3"],
        "Map2+3": ["Map2", "Map3"],
        "Map1+2+3": ["Map1", "Map2", "Map3"],
    }
    preprocessing_options = [
        [],  # No preprocessing
        ["EMSC"],
        ["Normalization"],
        ["SNV"],
        ["Second Derivative"],
        ["EMSC", "Normalization"],
        ["EMSC", "SNV"],
        ["EMSC", "Second Derivative"],
    ]

    results = []
    for preprocess_methods in preprocessing_options:
        preprocess_label = "+".join(preprocess_methods) if preprocess_methods else "None"

        for label, maps in map_sets.items():
            print(f"\n🧪 Evaluating {label} with preprocessing: {preprocess_label}")
            X_full, _ = aggregate_mapping_spectra(data_dir, mapping_structure, maps, wavenumbers=wavenumbers)

            sample_ids = [
                sid
                for sid, maps_all in mapping_structure.items()
                if all(m in maps_all for m in maps)
            ]
            sample_ids = [sid for sid in sample_ids if sid in y_df.index]
            y_clean = y_df.loc[sample_ids, target]

            y_parts = []
            for sid in sample_ids:
                if pd.isna(y_clean.get(sid)):
                    continue
                count = sum(len(mapping_structure[sid][m]) * 3 for m in maps)
                if count > 0:
                    y_parts.append(np.repeat(y_clean[sid], count))

            if not y_parts:
                print(f"Skipping {label} because no spectra match labels.")
                continue

            y = np.concatenate(y_parts)
            if X_full.shape[1] > len(y):
                X_full = X_full[:, : len(y)]
            elif X_full.shape[1] < len(y):
                print(f"[ERROR] More labels than spectra. Skipping {label}.")
                continue

            X_mat = X_full[:, : len(y)].T
            y_vec = y

            r2, rmse, y_pred, preds = run_mccv_for_mapping(X_mat, y_vec, preprocess_methods=preprocess_methods)

            # ─── Mean vs STD ───
            mean_preds = np.nanmean(preds, axis=1)
            std_preds  = np.nanstd(preds, axis=1)
        
            # build one sample‐ID per spectrum (you already have this)
            sids_expanded = []
            for sid in sample_ids:
                cnt = sum(len(mapping_structure[sid][m]) * 3 for m in maps)
                sids_expanded.extend([sid] * cnt)
        
            # — now collapse to one row per sample —

            df_stats = pd.DataFrame({
                'sid':        sids_expanded,
                'mean_pred':  mean_preds,
                'std_pred':   std_preds
            })
            # … your sample_stats and sample_colors code above …
            # collapse to one row per sample
            sample_stats = df_stats.groupby('sid').agg({
                'mean_pred':'mean',
                'std_pred':'mean'
            }).reset_index()
        
            # pick colors per sample (blue=train, red=test, black otherwise)
            if "Results_Combined" in output_dir:
                train_ids = set(pd.read_csv(train_meta_csv)["Sample_ID"])
                sample_colors = [
                    'blue' if sid in train_ids else 'red'
                    for sid in sample_stats['sid']
                ]
            else:
                sample_colors = 'black'
        
            # decide TRAIN/TEST/COMBINED label
            if "Results_TrainOnly" in output_dir:
                ds_label = "TRAIN"
            elif "Results_TestOnly" in output_dir:
                ds_label = "TEST"
            elif "Results_Combined" in output_dir:
                ds_label = "COMBINED"
            else:
                ds_label = os.path.basename(output_dir)
        
            # build save path
            std_plot_path = os.path.join(
                output_dir,
                f"Mean_STD_{label.replace('+','_')}_{preprocess_label}.png"
            )
        
            # plot Mean vs STD
            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(
                sample_stats['mean_pred'],
                sample_stats['std_pred'],
                c=sample_colors,
                edgecolor='k',
                alpha=0.5
            )
            cutoff = sample_stats['std_pred'].mean() + 3 * sample_stats['std_pred'].std()
            ax.axhline(cutoff, linestyle='--', color='gray', label='3σ cutoff')
            ax.text(
                0.05, 0.95,
                f"R² = {r2:.3f}\nRMSE = {rmse:.3f}",
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.7)
            )
            # SAVE OUTLIERS (samples with std_pred > 3σ)
            outliers = sample_stats[sample_stats['std_pred'] > cutoff]
            outliers.to_excel(os.path.join(output_dir, "Outliers.xlsx"), index=False)

            ax.set_xlabel("Mean Prediction (per sample)")
            ax.set_ylabel("Prediction STD (per sample)")
            ax.set_title(f"{label} — {ds_label} — Mean vs STD\n{preprocess_label}")
            plt.tight_layout()
            plt.savefig(std_plot_path, dpi=300, bbox_inches='tight')
            # plt.show() This shows every single STD vs mean plot
            plt.close()
        





            print(f"{label} ({preprocess_label}) — R²: {r2:.4f}, RMSE: {rmse:.4f}")
            results.append({"Mapping": label, "Preprocessing": preprocess_label, "R2": r2, "RMSE": rmse})

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_vec, y_pred, alpha=0.5, edgecolor="k")
            ax.plot([min(y_vec), max(y_vec)], [min(y_vec), max(y_vec)], "r--", linewidth=1.5)
            ax.set_xlabel("True Values", fontsize=12, fontweight="bold")
            ax.set_ylabel("Predicted Values", fontsize=12, fontweight="bold")
            ax.set_title(f"{label} Parity Plot\n{preprocess_label}\nR² = {r2:.3f}, RMSE = {rmse:.2f}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            parity_plot_path = os.path.join(output_dir, f"ParityPlot_{label.replace('+','_')}_{preprocess_label}.png")
            plt.savefig(parity_plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            residuals = y_pred - y_vec
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(residuals, bins=30, color="gray", edgecolor="black", alpha=0.7)
            ax.axvline(0, color="red", linestyle="--")
            ax.set_title(f"{label} Residuals Histogram\n{preprocess_label}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Prediction Error (y_pred - y_true)", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            plt.tight_layout()
            hist_path = os.path.join(output_dir, f"ResidualsHist_{label.replace('+','_')}_{preprocess_label}.png")
            plt.savefig(hist_path, dpi=300, bbox_inches="tight")
            plt.close()

    return pd.DataFrame(results)


def compute_average_baseline_per_mapping(data_dir, mapping_structure, wavenumbers):
    color_map = {"Map1": "blue", "Map2": "green", "Map3": "red"}
    baseline_data = {"Map1": [], "Map2": [], "Map3": []}

    for sample_id, maps in mapping_structure.items():
        for map_label in ["Map1", "Map2", "Map3"]:
            if map_label in maps:
                for file_path in maps[map_label]:
                    spectra = load_txt_file(file_path)
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

# ──────────────────────────────────────────────────────────
# ── “One‐time” overlay plots: run these exactly once at the top ──
# ──────────────────────────────────────────────────────────

train_data_dir = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\data"
train_meta_csv  = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\y_metadata.csv"

# Load wavenumbers & build mapping structure
wavenumbers, _, _       = load_data(train_data_dir)
mapping_structure       = parse_mapping_files(train_data_dir)

# ── Replace “overlay_base” with your Per_Sample_Overlay_Plots folder:
overlay_base = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Per_Sample_Overlay_Plots"
os.makedirs(overlay_base, exist_ok=True)

# 1) Per‐sample overlays (Map1, Map2, Map3) → saved into Per_Sample_Overlay_Plots/
plot_individual_mapping_overlay(
    wavenumbers,
    mapping_structure,
    train_data_dir,
    overlay_base
)

# 2) “All maps combined” overlay → one file Overlay_All_Mappings.png in Per_Sample_Overlay_Plots/
full_spectra, color_labels = aggregate_mapping_spectra(
    train_data_dir,
    mapping_structure,
    ["Map1", "Map2", "Map3"],
    wavenumbers=wavenumbers
)
plot_all_mappings(
    wavenumbers,
    full_spectra,
    color_labels,
    os.path.join(overlay_base, "Overlay_All_Mappings.png")
)

# 3) “Each map separately” overlays → Overlay_Map1_Only.png, Overlay_Map2_Only.png, Overlay_Map3_Only.png
plot_separated_mappings(
    wavenumbers,
    full_spectra,
    color_labels,
    overlay_base
)

# ──────────────────────────────────────────────────────────
# ── Now run the MCCV pipeline (which no longer re‐plots overlays) ──
# ──────────────────────────────────────────────────────────
TRAIN_OUT = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Results_TrainOnly"
TEST_OUT  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Results_TestOnly"
COMB_OUT  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Results_Combined"
# 1) On TRAIN only:
results_train_only = run_mapping_comparison(
    data_dir      = train_data_dir,
    metadata_path = train_meta_csv,
    output_dir    = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Results_TrainOnly"
)
results_train_only.to_excel(
    r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Results_TrainOnly\Results_TrainOnly.xlsx",
    index=False
)
# --- INSERT THIS: show top 3 TRAIN plots ---
top3 = results_train_only.sort_values("R2", ascending=False).head(3)
for _, row in top3.iterrows():
    m = row["Mapping"].replace("+","_")
    p = row["Preprocessing"]
    img_path = os.path.join(TRAIN_OUT, f"Mean_STD_{m}_{p}.png")
    img = plt.imread(img_path)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{row['Mapping']} | {p} | R²={row['R2']:.3f}")
    plt.show()

# 2) On TEST only:
test_data_dir = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\data_test"
test_meta_csv = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata_test_updated.csv"

results_test_only = run_mapping_comparison(
    data_dir      = test_data_dir,
    metadata_path = test_meta_csv,
    output_dir    = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Results_TestOnly"
)
results_test_only.to_excel(
    r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Results_TestOnly\Results_TestOnly.xlsx",
    index=False
)
# --- INSERT THIS: show top 3 TEST plots ---
top3 = results_test_only.sort_values("R2", ascending=False).head(3)
for _, row in top3.iterrows():
    m = row["Mapping"].replace("+","_")
    p = row["Preprocessing"]
    img_path = os.path.join(TEST_OUT, f"Mean_STD_{m}_{p}.png")
    img = plt.imread(img_path)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{row['Mapping']} | {p} | R²={row['R2']:.3f}")
    plt.show()
# 3) Combined run:
combined_data_dir = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Combined_Data"
combined_meta_csv = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Combined_Metadata.csv"

results_combined = run_mapping_comparison(
    data_dir      = combined_data_dir,
    metadata_path = combined_meta_csv,
    output_dir    = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Results_Combined"
)
results_combined.to_excel(
    r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Results_Combined\Results_Combined.xlsx",
    index=False
)
# --- INSERT THIS: show top 3 COMBINED plots ---
top3 = results_combined.sort_values("R2", ascending=False).head(3)
for _, row in top3.iterrows():
    m = row["Mapping"].replace("+","_")
    p = row["Preprocessing"]
    img_path = os.path.join(COMB_OUT, f"Mean_STD_{m}_{p}.png")
    img = plt.imread(img_path)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{row['Mapping']} | {p} | R²={row['R2']:.3f}")
    plt.show()

# ──────────────────────────────────────────────────────────
# ── Combined run: copy both train & test into a new folder, concatenate CSVs ──
# ──────────────────────────────────────────────────────────

# a) Create a “combined_data_dir” and copy all .txt files from train and test into it:
combined_data_dir = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Combined_Data"
os.makedirs(combined_data_dir, exist_ok=True)

# Copy every .txt from train_data_dir → combined_data_dir
for fname in os.listdir(train_data_dir):
    if fname.lower().endswith(".txt"):
        src = os.path.join(train_data_dir, fname)
        dst = os.path.join(combined_data_dir, fname)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            from shutil import copy2
            copy2(src, dst)

# Copy every .txt from test_data_dir → combined_data_dir
for fname in os.listdir(test_data_dir):
    if fname.lower().endswith(".txt"):
        src = os.path.join(test_data_dir, fname)
        dst = os.path.join(combined_data_dir, fname)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            from shutil import copy2
            copy2(src, dst)

# b) Read both metadata CSVs, concatenate them, and save to a new CSV:
combined_meta_csv = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Combined_Metadata.csv"

df_train = pd.read_csv(train_meta_csv)
df_test  = pd.read_csv(test_meta_csv)

# Make sure indices match “Sample_ID” format if needed; if your metadata files already have a “Sample_ID” column:
# (Otherwise adjust to whatever key your run_mapping_comparison expects.)
df_combined = pd.concat([df_train, df_test], ignore_index=True)
df_combined.to_csv(combined_meta_csv, index=False)

# c) Finally, call run_mapping_comparison on the new combined folder + new combined CSV:
results_combined = run_mapping_comparison(
    data_dir      = combined_data_dir,
    metadata_path = combined_meta_csv,
    output_dir    = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison\Results_Combined"
)
