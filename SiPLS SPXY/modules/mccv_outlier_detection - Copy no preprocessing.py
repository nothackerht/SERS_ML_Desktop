import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from modules.data_loader import load_data, load_metadata
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

def run_mccv_outlier_detection(
    train_data_dir,
    train_meta_path,
    test_data_dir,
    test_meta_path,
    output_dir=r"C:\Users\spect\Desktop\MD-Analysis-main (3)\MCCV results",
    target_columns=['target_SI', 'HGS_pp_avg', 'ADF_pp_avg'],
    n_iter=500,
    test_size=0.2,
    random_state=42
):
    os.makedirs(output_dir, exist_ok=True)

    _, _, X_train = load_data(train_data_dir)
    _, _, X_test = load_data(test_data_dir)
    X_all = np.hstack([X_train, X_test]).T

    y_train = load_metadata(train_meta_path)
    y_test = load_metadata(test_meta_path)
    y_all = pd.concat([y_train, y_test], ignore_index=True)
    sample_ids = y_all['SampleID'].values
    source_labels = ['Train'] * len(y_train) + ['Test'] * len(y_test)

    for col in target_columns:
        if col not in y_all.columns:
            print(f"Skipping {col}, not found.")
            continue

        sample_mask = ~y_all[col].isna()
        y_valid = y_all[col].values[sample_mask]
        ids_valid = sample_ids[sample_mask]
        src_valid = np.array(source_labels)[sample_mask]

        spectral_mask = np.repeat(sample_mask.values, 9)
        X_valid = X_all[spectral_mask]

        n_samples = len(y_valid)
        spectra_per_sample = 9
        preds = np.full((n_samples, n_iter), np.nan)

        print(f"Running MCCV for target: {col} on {n_samples} samples ({n_samples * spectra_per_sample} spectra)...")

        for i in range(n_iter):
            try:
                sample_indices = np.arange(n_samples)
                train_sample_idx, test_sample_idx = train_test_split(
                    sample_indices, test_size=test_size, random_state=random_state + i
                )
                train_spectrum_idx = np.hstack([
                    np.arange(s * spectra_per_sample, (s + 1) * spectra_per_sample) for s in train_sample_idx
                ])
                test_spectrum_idx = np.hstack([
                    np.arange(s * spectra_per_sample, (s + 1) * spectra_per_sample) for s in test_sample_idx
                ])

                X_train_iter = X_valid[train_spectrum_idx]
                y_train_iter = np.repeat(y_valid[train_sample_idx], spectra_per_sample)
                X_test_iter = X_valid[test_spectrum_idx]

                pls = PLSRegression(n_components=5)
                pls.fit(X_train_iter, y_train_iter)
                y_pred = pls.predict(X_test_iter).flatten()

                for idx, sample_idx in enumerate(test_sample_idx):
                    start = idx * spectra_per_sample
                    end = (idx + 1) * spectra_per_sample
                    preds[sample_idx, i] = np.mean(y_pred[start:end])
            except Exception as e:
                print(f"Iteration {i} failed: {e}")

        mean_pred = np.nanmean(preds, axis=1)
        std_pred = np.nanstd(preds, axis=1)
        abs_error = np.abs(y_valid - mean_pred)
        outlier_std = std_pred > 0.07
        outlier_error = abs_error > 0.3
        outlier_both = outlier_std & outlier_error

        # Save residual histogram
        plt.figure(figsize=(8, 6))
        plt.hist(abs_error, bins=20, color='gray', edgecolor='black')
        plt.axvline(0.3, color='red', linestyle='--', label='Error Threshold (0.3)')
        plt.xlabel("Absolute Prediction Error")
        plt.ylabel("Frequency")
        plt.title(f"Residual Histogram: {col}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Histogram_Abs_Error_{col}.png"), dpi=300)
        plt.close()

        # Save MEAN-STD Plot
        plt.figure(figsize=(8, 6))
        for source in ['Train', 'Test']:
            mask = src_valid == source
            color = 'blue' if source == 'Train' else 'red'
            plt.scatter(mean_pred[mask], std_pred[mask], label=source, color=color, alpha=0.6, edgecolor='k')
        for i in range(n_samples):
            plt.text(mean_pred[i], std_pred[i], str(ids_valid[i]), fontsize=6, alpha=0.6)
        plt.xlabel("Mean Prediction")
        plt.ylabel("Prediction STD")
        plt.title(f"MCCV MEAN-STD: {col}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"MCCV_MEAN_STD_{col}.png"), dpi=300)
        plt.close()

        # Create and save Excel log
        df_log = pd.DataFrame({
            'SampleID': ids_valid,
            'Source': src_valid,
            'True': y_valid,
            'Mean Prediction': mean_pred,
            'STD Prediction': std_pred,
            'Abs Error': abs_error,
            'Outlier_STD': outlier_std,
            'Outlier_Error': outlier_error,
            'Outlier_Both': outlier_both
        })
        df_log.sort_values(by='Abs Error', ascending=False, inplace=True)

        excel_path = os.path.join(output_dir, f"MCCV_Outliers_{col}.xlsx")
        df_log.to_excel(excel_path, index=False)

        # Add Excel highlighting
        wb = load_workbook(excel_path)
        ws = wb.active
        std_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        error_fill = PatternFill(start_color="FF9999", end_color="FF9999", fill_type="solid")
        both_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")

        for row in range(2, ws.max_row + 1):
            if ws[f"K{row}"].value:  # Outlier_Both
                ws[f"K{row}"].fill = both_fill
            elif ws[f"I{row}"].value:  # Outlier_STD
                ws[f"I{row}"].fill = std_fill
            elif ws[f"J{row}"].value:  # Outlier_Error
                ws[f"J{row}"].fill = error_fill

        wb.save(excel_path)

        print(f"✓ Finished: {col} — Saved Excel, Histogram, and Mean-STD Plot.")
