# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:59:02 2025

@author: spect
"""

# This script adds a new method to RegressionModel that uses SPXY to split the training set
# for SiPLS hyperparameter optimization and blind test set evaluation (without retraining on full training data)

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt
from modules.preprocessing import Preprocessing
from modules.data_loader import load_data, load_metadata
from scipy.spatial.distance import cdist
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
import seaborn as sns
from itertools import product  # needed for shift logic

def spxy_split(X, y, ratio=0.8):
    num_samples = X.shape[0] // 9
    X_avg = np.mean(X.reshape(num_samples, 9, -1), axis=1)
    y_avg = np.mean(y.reshape(num_samples, 9, -1), axis=1)

    D_X = cdist(X_avg, X_avg, metric='euclidean')
    D_y = cdist(y_avg, y_avg, metric='euclidean')
    D = D_X + D_y

    selected = [np.unravel_index(np.argmax(D), D.shape)[0]]
    while len(selected) < int(num_samples * ratio):
        remaining = list(set(range(num_samples)) - set(selected))
        min_dist = [min(D[i, selected]) for i in remaining]
        selected.append(remaining[np.argmax(min_dist)])

    selected = np.array(selected)
    all_indices = np.arange(num_samples)
    train_indices = selected
    val_indices = np.setdiff1d(all_indices, selected)
    return train_indices, val_indices

class RegressionModel:
    def __init__(self, all_spectra, y_label_df, wavenumbers=None):
        self.all_spectra = all_spectra
        self.y_label_df = y_label_df
        self.wavenumbers = wavenumbers

    def preprocess_and_filter_data(self, preprocess_methods=[]):
        if preprocess_methods:
            preprocess = Preprocessing(self.all_spectra)
            x_spectra = preprocess.preprocess(preprocess_methods)
        else:
            x_spectra = self.all_spectra.copy()

    
        control_dm1_indices = self.y_label_df['Type'].isin(['Control', 'DM1'])
        y_label_filtered_df = self.y_label_df[control_dm1_indices].reset_index(drop=True)
    
        y_label_filtered_df.loc[y_label_filtered_df['Type'] == 'Control', 'target_SI'] = 0
        y_label_filtered_df.loc[y_label_filtered_df['Type'] == 'Control', 'HGS_pp_avg'] = 100
        y_label_filtered_df.loc[y_label_filtered_df['Type'] == 'Control', 'ADF_pp_avg'] = 100
    
        valid_samples = y_label_filtered_df.dropna(subset=['ADF_pp_avg']).index
        y_label_filtered_df = y_label_filtered_df.loc[valid_samples].reset_index(drop=True)
    
        y_target_SI = y_label_filtered_df['target_SI'].values
        y_HGS = y_label_filtered_df['HGS_pp_avg'].values
        y_ADF = y_label_filtered_df['ADF_pp_avg'].values
        y_combined = np.vstack([y_target_SI, y_HGS, y_ADF]).T
    
        valid_spectra_indices = np.concatenate([np.arange(i * 9, (i + 1) * 9) for i in valid_samples])
        x_filtered = x_spectra[:, valid_spectra_indices]
    
        y_combined_repeated = np.repeat(y_combined, 9, axis=0)
    
        return x_filtered, y_combined_repeated
    
    def sipls_single_run_with_log(self, X_train, y_train, X_test, y_test,
                                   max_components=10, num_intervals=10, max_combination_size=4,
                                   wavenumbers=None):
        split_indices = np.linspace(0, X_train.shape[1], num_intervals + 1, dtype=int)
        interval_bounds = [(split_indices[i], split_indices[i + 1]) for i in range(num_intervals)]
        num_samples = X_train.shape[0] // 9
        groups = np.repeat(np.arange(num_samples), 9)
        group_kf = GroupKFold(n_splits=5)

        def evaluate_combo_with_shifts(base_combo):
            best_shift_rmse = np.inf
            best_result = None

            for shift_pattern in product([-1, 0, 1], repeat=len(base_combo) * 2):
                shifted_combo = []
                for i, (start, end) in enumerate(base_combo):
                    shift_start = shift_pattern[2 * i]
                    shift_end = shift_pattern[2 * i + 1]
                    start_idx = max(start + shift_start, 0)
                    end_idx = min(end + shift_end, X_train.shape[1])
                    if end_idx <= start_idx:
                        break
                    shifted_combo.append((start_idx, end_idx))
                else:
                    current_cols = np.hstack([np.arange(start, end) for (start, end) in shifted_combo])
                    if len(current_cols) < max_components:
                        continue

                    X_train_combo = X_train[:, current_cols]
                    X_test_combo = X_test[:, current_cols]
                    fold_rmses = []
                    for train_idx, val_idx in group_kf.split(X_train_combo, y_train, groups=groups):
                        X_train_cv, X_val_cv = X_train_combo[train_idx], X_train_combo[val_idx]
                        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
                        pls = PLSRegression(n_components=max_components)
                        pls.fit(X_train_cv, y_train_cv)
                        y_val_pred = pls.predict(X_val_cv)
                        rmse = mean_squared_error(y_val_cv, y_val_pred, squared=False)
                        fold_rmses.append(rmse)

                    avg_rmse = np.mean(fold_rmses)
                    if avg_rmse < best_shift_rmse:
                        pls = PLSRegression(n_components=max_components)
                        pls.fit(X_train_combo, y_train)
                        y_pred_test = pls.predict(X_test_combo)
                        wavenumber_ranges = [
                            (wavenumbers[start], wavenumbers[end - 1]) if wavenumbers is not None else (start, end - 1)
                            for (start, end) in shifted_combo
                        ]
                        best_result = {
                            'combo': shifted_combo,
                            'rmse': avg_rmse,
                            'y_pred': y_pred_test,
                            'wavenumber_ranges': wavenumber_ranges
                        }
                        best_shift_rmse = avg_rmse

            return best_result

        all_combos = []
        for r in range(1, max_combination_size + 1):
            all_combos.extend(list(combinations(interval_bounds, r)))

        self.current_preprocessing = getattr(self, 'current_preprocessing', 'Not Specified')

        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(evaluate_combo_with_shifts)(combo) for combo in all_combos
        )

        results = [r for r in results if r is not None and 'wavenumber_ranges' in r]
        best_result = min(results, key=lambda x: x['rmse'])

        interval_rmse_log = [
            {
                'Base Combo': str(r['combo']),
                'Final Combo': str(r['combo']),
                'Wavenumber Ranges': r['wavenumber_ranges'],
                'Average RMSE': r['rmse']
            } for r in results
        ]

        return best_result['y_pred'], best_result['wavenumber_ranges'], [], interval_rmse_log



    def evaluate_test_set_with_sipls_spxy(self, test_spectra_path, test_metadata_path,
                                          preprocess_methods=[],
                                          n_components_list=[2, 3],
                                          num_intervals_list=[10],
                                          max_combination_size=4,
                                          output_dir=r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SiPLS SPXY\SPXY RESULTS"):
    
        os.makedirs(output_dir, exist_ok=True)
    
        X_full, y_full = self.preprocess_and_filter_data(preprocess_methods)
        X = X_full.T
        sample_indices = np.arange(y_full.shape[0] // 9)
        spectra_indices = np.repeat(sample_indices, 9)
    
        train_idx, val_idx = spxy_split(X, y_full)
        train_spectra_idx = np.isin(spectra_indices, train_idx)
        val_spectra_idx = np.isin(spectra_indices, val_idx)
        # Log SPXY train/val sample indices and SampleIDs
        valid_sample_mask = self.y_label_df['Type'].isin(['Control', 'DM1'])
        valid_sample_df = self.y_label_df[valid_sample_mask].reset_index(drop=True)
        
        train_ids = valid_sample_df.loc[train_idx, 'SampleID'].values
        val_ids = valid_sample_df.loc[val_idx, 'SampleID'].values
        
        split_log_path = os.path.join(output_dir, "SPXY_Split_Indices.xlsx")
        split_log_df = pd.DataFrame({
            'Train Sample Index': pd.Series(train_idx),
            'Train SampleID': pd.Series(train_ids),
            'Validation Sample Index': pd.Series(val_idx),
            'Validation SampleID': pd.Series(val_ids)
        })
        split_log_df.to_excel(split_log_path, index=False)


        X_train, y_train = X[train_spectra_idx], y_full[train_spectra_idx]
        # === OPTIONAL: Visualize SPXY split on target values ===
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_style("whitegrid")
        
        target_names = ['target_SI', 'HGS_pp_avg', 'ADF_pp_avg']
        y_full_avg = y_full[::9]  # Averaged back to one sample per 9 spectra
        train_target_vals = y_full_avg[train_idx]
        val_target_vals = y_full_avg[val_idx]
        
        for i, target in enumerate(target_names):
            train_vals = train_target_vals[:, i]
            val_vals = val_target_vals[:, i]
        
            if np.all(np.isnan(train_vals)) or np.all(np.isnan(val_vals)):
                print(f"Skipping SPXY split plot for {target} — all values are NaN.")
                continue
        
            # Scatter plot by sample index
            plt.figure(figsize=(8, 4))
            plt.scatter(train_idx, train_vals, label='SPXY Train', color='blue')
            plt.scatter(val_idx, val_vals, label='SPXY Val', color='orange')
            plt.xlabel("Sample Index")
            plt.ylabel(f"{target}")
            plt.title(f"SPXY Split: {target}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"SPXY_Split_{target}_Scatter.png"), dpi=300)
            plt.close()
        
            # Boxplot
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=[train_vals, val_vals], palette=['blue', 'orange'])
            plt.xticks([0, 1], ['Train', 'Validation'])
            plt.ylabel(target)
            plt.title(f"SPXY Split Distribution: {target}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"SPXY_Split_{target}_Boxplot.png"), dpi=300)
            plt.close()
        
            # KDE
            plt.figure(figsize=(6, 4))
            sns.kdeplot(train_vals[~np.isnan(train_vals)], label='Train', fill=True, color='blue')
            sns.kdeplot(val_vals[~np.isnan(val_vals)], label='Validation', fill=True, color='orange')
            plt.xlabel(target)
            plt.title(f"SPXY Split KDE: {target}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"SPXY_Split_{target}_KDE.png"), dpi=300)
            plt.close()

        X_val, y_val = X[val_spectra_idx], y_full[val_spectra_idx]
    
        best_rmse = np.inf
        best_params = None
        best_combo = None
        # Track current preprocessing method for logging
        self.current_preprocessing = ' + '.join(preprocess_methods) if preprocess_methods else 'None'

   

        for n_comp in n_components_list:
            for n_intervals in num_intervals_list:
                print("=" * 80)
                print(f"Starting combination: Preprocessing = {self.current_preprocessing}, Components = {n_comp}, Intervals = {n_intervals}")
                print("=" * 80)
                y_val_pred, interval_combo, _, _ = self.sipls_single_run_with_log(
                    X_train, y_train, X_val, y_val,
                    max_components=n_comp,
                    num_intervals=n_intervals,
                    max_combination_size=max_combination_size,
                    wavenumbers=self.wavenumbers
                )

                rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = (n_comp, n_intervals)
                    best_combo = interval_combo
                    # After both loops finish
                    if best_val_pred is None:
                        raise RuntimeError("No valid SiPLS model was found. Check data, intervals, and preprocessing.")

                    # Generate shifted versions of best combo (±1 for each bound)
                    from itertools import product
                    def shift_best_combo_only(best_combo, max_shift=1, max_wavenumber_idx=None):
                        shifted_combos = []
                        for shift_pattern in product(range(-max_shift, max_shift + 1), repeat=2 * len(best_combo)):
                            shifted = []
                            for i, (start, end) in enumerate(best_combo):
                                shift_start = shift_pattern[2 * i]
                                shift_end = shift_pattern[2 * i + 1]
                                new_start = max(0, start + shift_start)
                                new_end = min(max_wavenumber_idx, end + shift_end) if max_wavenumber_idx else end + shift_end
                                if new_end <= new_start:
                                    break
                                shifted.append((new_start, new_end))
                            else:
                                shifted_combos.append(shifted)
                        return shifted_combos
                    
                    # Determine max index based on number of wavenumbers
                    max_wavenumber_idx = len(self.wavenumbers) - 1
                    shifted_best_combos = shift_best_combo_only(best_combo, max_shift=1, max_wavenumber_idx=max_wavenumber_idx)
                    # Evaluate each shifted combo on validation set to find the best one
                    best_shifted_rmse = np.inf
                    for combo in shifted_best_combos:
                        cols = []
                        for start_idx, end_idx in combo:
                            cols.extend(list(range(start_idx, end_idx)))
                        X_train_shift = X_train[:, cols]
                        X_val_shift = X_val[:, cols]
                    
                        pls = PLSRegression(n_components=best_params[0])
                        pls.fit(X_train_shift, y_train)
                        val_pred_shift = pls.predict(X_val_shift)
                        rmse_shift = np.sqrt(mean_squared_error(y_val, val_pred_shift))
                    
                        if rmse_shift < best_shifted_rmse:
                            best_combo = [(self.wavenumbers[start_idx], self.wavenumbers[end_idx - 1]) for start_idx, end_idx in combo]
                            best_val_pred = val_pred_shift
                            best_shifted_rmse = rmse_shift

                    
                    best_val_true = y_val

    
        print("Best on SPXY 20% validation:", best_params, best_rmse)
        # === Evaluate SPXY validation performance ===
        num_val_samples = best_val_true.shape[0] // 9
        avg_val_true = np.mean(best_val_true.reshape(num_val_samples, 9, -1), axis=1)
        avg_val_pred = np.mean(best_val_pred.reshape(num_val_samples, 9, -1), axis=1)
        val_r2 = r2_score(avg_val_true, avg_val_pred)
        val_rmse = np.sqrt(mean_squared_error(avg_val_true, avg_val_pred))
        print(f"SPXY Validation R²: {val_r2:.3f}, RMSE: {val_rmse:.3f}")

        # Save best hyperparameters and interval combo
        pd.DataFrame([{
            'Best N Components': best_params[0],
            'Best Num Intervals': best_params[1],
            'Validation RMSE': best_rmse,
            'Validation R²': val_r2,
            'Selected Intervals': best_combo
        }]).to_excel(os.path.join(output_dir, "Best_SiPLS_SPXY_Settings.xlsx"), index=False)
        
            
        # === Evaluate on test set using model trained only on SPXY training subset ===
        X_test_wavs, _, X_test_raw = load_data(test_spectra_path)
        test_meta = load_metadata(test_metadata_path)
    
        test_proc = X_test_raw.copy()
        if preprocess_methods:
            for method in preprocess_methods:
                if method == 'EMSC':
                    emsc_ref = np.mean(X_train.T, axis=1)
                    test_proc = Preprocessing(None).emsc(test_proc, reference=emsc_ref)
                else:
                    test_proc = Preprocessing(test_proc).preprocess([method])
        X_test = test_proc.T
        
            
        indices = []
        for start_cm, end_cm in best_combo:
            start = np.argmin(np.abs(np.array(self.wavenumbers) - start_cm))
            end = np.argmin(np.abs(np.array(self.wavenumbers) - end_cm)) + 1
            indices.extend(list(range(start, end)))
    
        X_train_best = X_train[:, indices]
        X_test_best = X_test[:, indices]
    
        avg_preds = {}
        for i, target in enumerate(['target_SI', 'HGS_pp_avg', 'ADF_pp_avg']):
            if target not in test_meta.columns or test_meta[target].dropna().empty:
                print(f"Skipping {target} — no values found in test metadata.")
                continue
    
            y_train_target = y_train[:, i].reshape(-1, 1)
            pls = PLSRegression(n_components=best_params[0])
            pls.fit(X_train_best, y_train_target)
            
            # === Evaluate Training Performance ===
            y_train_pred = pls.predict(X_train_best)
            avg_train_pred = np.mean(y_train_pred.reshape(-1, 9), axis=1)
            avg_train_true = np.mean(y_train_target.reshape(-1, 9), axis=1)
            train_r2 = r2_score(avg_train_true, avg_train_pred)
            train_rmse = np.sqrt(mean_squared_error(avg_train_true, avg_train_pred))
            print(f"{target} — Train R²: {train_r2:.3f}, RMSE: {train_rmse:.3f}")
            
            # === Predict on Test Set ===
            y_test_pred = pls.predict(X_test_best)

    
            num_samples = y_test_pred.shape[0] // 9
            avg_pred = np.mean(y_test_pred.reshape(num_samples, 9), axis=1)
            std_pred = np.std(y_test_pred.reshape(num_samples, 9), axis=1)
            true_vals = test_meta[target].values[:num_samples]
            mask = ~np.isnan(true_vals)
            avg_pred, true_vals, std_pred = avg_pred[mask], true_vals[mask], std_pred[mask]
    
            if len(true_vals) == 0:
                print(f"Skipping {target} — all values are NaN.")
                continue
    
            r2 = r2_score(true_vals, avg_pred)
            rmse = np.sqrt(mean_squared_error(true_vals, avg_pred))
            avg_preds[target] = {
                'Train R²': train_r2,
                'Train RMSE': train_rmse,
                'Test R²': r2,
                'Test RMSE': rmse
            }

            print(f"{target}: R²={r2:.3f}, RMSE={rmse:.3f}")
    
            # Save parity plot
            plt.figure(figsize=(6, 6))
            plt.errorbar(true_vals, avg_pred, yerr=std_pred, fmt='o', capsize=4, color='blue', ecolor='gray')
            plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--')
            plt.xlabel(f"Actual {target}", fontsize=12, fontweight='bold')
            plt.ylabel(f"Predicted {target}", fontsize=12, fontweight='bold')
            plt.title(f"{target} | SiPLS SPXY | C={best_params[0]}, I={best_params[1]}\nR² = {r2:.3f}", fontsize=14, fontweight='bold')
            plt.tight_layout()
    
            fname = f"SPXY_Parity_{target}_C{best_params[0]}_I{best_params[1]}.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=300)
            plt.close()
    
            # Save predictions
            pd.DataFrame({
                'True': true_vals,
                'Predicted': avg_pred
            }).to_excel(os.path.join(output_dir, f"SPXY_Results_{target}.xlsx"), index=False)
    
            # Plot selected intervals on spectra for this target
            self.plot_selected_sipls_intervals(
                selected_intervals=best_combo,
                output_dir=output_dir,
                title=f"{target} | SiPLS SPXY Best Model",
                spectra_for_overlay=X_train.T,
                target_label=target,
                n_components=best_params[0],
                num_intervals=best_params[1],
                max_combo_size=max_combination_size,
                preprocessing=self.current_preprocessing,
                save_plot=True,
                show_plot=True
            )
        # Save final summary
        pd.DataFrame(avg_preds).T.to_excel(os.path.join(output_dir, "SPXY_SiPLS_Test_Performance.xlsx"))

    def plot_selected_sipls_intervals(self, selected_intervals, output_dir, title,
                                      save_plot=True, show_plot=True, plot_average=True,
                                      spectra_for_overlay=None, target_label=None,
                                      n_components=None, num_intervals=None, max_combo_size=None,
                                      preprocessing=None):
    
        import matplotlib.pyplot as plt
    
        if self.wavenumbers is None:
            print("Wavenumbers not provided. Skipping interval plot.")
            return
    
        fig, ax = plt.subplots(figsize=(10, 2.5))
    
        # Overlay spectra used for training
        if spectra_for_overlay is not None:
            for i in range(spectra_for_overlay.shape[1]):
                ax.plot(self.wavenumbers, spectra_for_overlay[:, i], color='gray', alpha=0.2, linewidth=0.5)
    
        # Highlight selected intervals
        for (start_cm, end_cm) in selected_intervals:
            ax.axvspan(start_cm, end_cm, color='skyblue', alpha=0.5)
    
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
        ax.set_ylabel("Intensity (a.u.)", fontsize=11)
    
        # Build informative title
        plot_title = title
        if target_label:
            plot_title += f"\nTarget: {target_label}"
        if n_components is not None and num_intervals is not None:
            plot_title += f" | C={n_components}, I={num_intervals}"
        if preprocessing:
            plot_title += f"\nPreproc: {preprocessing}"
    
        ax.set_title(plot_title, fontsize=12)
    
        plt.tight_layout()
    
        if save_plot:
            fname = f"Selected_SiPLS_Intervals_{target_label}_C{n_components}_I{num_intervals}.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()
