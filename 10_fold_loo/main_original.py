
"""

@author: Matthew Glace (glacemk@vcu.edu)

Virginia Commonwealth University, 2024

Data Analysis for Surface-Enhanced Raman Spectroscopy of Muscular Dystrophy Plasma Samples

"""


#%% Data Loading 

from modules.data_loader import load_data, load_metadata
# from modules.unsupervised_clustering import UnsupervisedClustering
from modules.plot_spectra import create_plot
# from modules.analysis_tools import run_analysis, run_ejcr_analysis, create_vip_plot
# from modules.substrate_evaluation import plot_spectra_with_baseline
# from modules.classification import Classifier
# from modules.SiPLS_Regression import RegressionModel
from modules.plot_spectra import plot_train_test_spectra
from modules.plot_spectra import plot_3d_target_metrics
from modules.preprocessing import Preprocessing
import matplotlib.pyplot as plt
# from modules.shell_evaluation import run_interval_selection_and_modeling



from modules.ten_fold_loo_shell_evaluation import leave_one_out_test_evaluation


import pandas as pd 
import numpy as np
import os
import torch
from itertools import product
print("CUDA Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Specify the directory containing the Data and Metadata
data_directory = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
meta_data_directory = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"
method_data_directory = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\method_data.csv"

output_dir = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\10_fold_results"
fold_log_dir = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\10_fold_results"


# Load the Data and Metadata
wavenumbers, averaged_spectra, all_spectra, filenames_per_column = load_data(
    data_directory, metadata_path=meta_data_directory, return_filenames=True
)


y_label_df = load_metadata(meta_data_directory)

# #%% Plot Averaged Spectral Data
# print("Wavenumbers shape:", wavenumbers.shape)     # Expect (1731,)
# print("All spectra shape:", all_spectra.shape)     # Expect (1731, N)

# # Plot and save the spectra
# create_plot(wavenumbers, all_spectra, y_label_df, f'{output_dir}/all_spectra_plot_combined.png', filenames_per_column)


# # Load training data
# train_data_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
# train_meta_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"
# wavs_train, _, spectra_train, train_filenames = load_data(train_data_path, metadata_path=train_meta_path, return_filenames=True)
# meta_train = load_metadata(train_meta_path)
# meta_train['Set'] = 'Train'

# # Load test data
# test_data_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data_test_updated"
# test_meta_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata_test_updated.csv"
# wavs_test, _, spectra_test, test_filenames = load_data(test_data_path, metadata_path=test_meta_path, return_filenames=True)
# meta_test = load_metadata(test_meta_path)
# meta_test['Set'] = 'Test'

# # Combine
# all_spectra_combined = np.hstack([spectra_train, spectra_test])
# meta_combined = pd.concat([meta_train, meta_test], axis=0).reset_index(drop=True)

# #%% Principal Component Analysis 

# #%% Principal Component Analysis 
# # Initialize clustering object with full combined metadata for the 3D baseline PCA plot
# clustering = UnsupervisedClustering(meta_combined, all_spectra_combined, output_dir)

# # Define all preprocessing combos
# preprocessing_combos = [
#     [],  # No preprocessing
#     ['Normalization'],
#     ['SNV'],
#     ['EMSC'],
#     ['Second Derivative'],
#     ['Normalization', 'SNV'],
#     ['EMSC', 'Second Derivative']
# ]

# # Add 3D PCA vs baseline visualization with SI coloring (only on raw spectra)
# clustering.pca_vs_baseline_3d_with_severity(spectra=clustering.spectra, save_plot=True)

# # ───────────── PCA on Train and Test Sets Separately (All Preprocessing Combos) ─────────────

# # Train set PCA
# clustering_train = UnsupervisedClustering(meta_train, spectra_train, output_dir)
# clustering_test = UnsupervisedClustering(meta_test, spectra_test, output_dir)

# for methods in preprocessing_combos:
#     name_part = '__'.join(methods) if methods else 'no_preprocessing'
#     custom_prefix = f"Train + Test PCA ({'+'.join(methods)})" if methods else "Train+Test PCA (No Preprocessing)"

#     # Train-only plots
#     clustering_train.perform_pca_and_plot(
#         x_array=spectra_train,
#         plot_name=f"pca_train_only_pc1_vs_pc2_{name_part}",
#         pc_axes=(1, 2),
#         save_plot=True,
#         preprocessing=methods,
#         custom_title_prefix="Train Set"
#     )
#     clustering_train.perform_pca_and_plot_with_severity(
#         x_array=spectra_train,
#         plot_name=f"pca_train_only_pc1_vs_pc2_severity_{name_part}",
#         pc_axes=(1, 2),
#         save_plot=True,
#         preprocessing=methods,
#         custom_title_prefix="Train Set"
#     )

#     # Fit PCA on training set
#     pca_model, _ = clustering_train.fit_pca_on_train(
#         x_train=spectra_train,
#         preprocessing=methods,
#         n_components=3
#     )
#     # Project test set using train PCA model
#     test_scores = pca_model.transform(Preprocessing(spectra_test).preprocess(methods=methods).T
#                                       if methods else spectra_test.T)

#     # Project train set using train PCA model
#     train_scores = pca_model.transform(Preprocessing(spectra_train).preprocess(methods=methods).T
#                                        if methods else spectra_train.T)

#     # Build metadata tables (repeat 9×)
#     meta_test_repeated = pd.DataFrame(np.repeat(meta_test.values, 9, axis=0), columns=meta_test.columns)
#     meta_train_repeated = pd.DataFrame(np.repeat(meta_train.values, 9, axis=0), columns=meta_train.columns)

#     meta_test_repeated['Set'] = 'Test'
#     meta_train_repeated['Set'] = 'Train'

#     # Combine and assign PCA scores
#     combined_meta = pd.concat([meta_train_repeated, meta_test_repeated], ignore_index=True)
#     combined_meta['PC1'] = np.concatenate([train_scores[:, 0], test_scores[:, 0]])
#     combined_meta['PC2'] = np.concatenate([train_scores[:, 1], test_scores[:, 1]])

#     import matplotlib.cm as cm
    
#     # Ensure target_SI exists
#     if 'target_SI' not in combined_meta.columns:
#         raise ValueError("Column 'target_SI' not found in metadata.")
    
#     fig, ax = plt.subplots(figsize=(16, 10))
    
#     # Build color gradient from SI values
#     scatter = ax.scatter(
#         combined_meta['PC1'],
#         combined_meta['PC2'],
#         c=combined_meta['target_SI'],
#         cmap='viridis',
#         s=25
#     )
    
#     # Add colorbar with label
#     cbar = plt.colorbar(scatter, ax=ax)
#     cbar.set_label('DM1 Severity (Splicing Index)', fontsize=18, fontweight='bold')
    
#     # Format plot
#     ax.set_xlabel("PC1", fontsize=24, fontweight='bold')
#     ax.set_ylabel("PC2", fontsize=24, fontweight='bold')
#     ax.set_title(f"Train + Test PCA by SI - {name_part}", fontsize=28, fontweight='bold')
    
#     # Save
#     plt.savefig(os.path.join(output_dir, f"pca_train_test_overlay_by_SI_{name_part}.png"), dpi=600, bbox_inches='tight')
#     plt.show()

#     ax.set_xlabel("PC1", fontsize=24, fontweight='bold')
#     ax.set_ylabel("PC2", fontsize=24, fontweight='bold')
#     ax.set_title(f"Train + Test PCA Projection - {name_part}", fontsize=28, fontweight='bold')
#     ax.legend(fontsize=18)
    
#     # Save figure
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, f"pca_train_test_overlay_pc1_vs_pc2_{name_part}.png"), dpi=600, bbox_inches='tight')
#     plt.show()

#     # Apply trained PCA model to test set (PC1 vs PC2)
#     clustering_test.transform_test_with_pca(
#         pca=pca_model,
#         x_test=spectra_test,
#         y_meta_test=meta_test.copy(),
#         preprocessing=methods,
#         pc_axes=(1, 2),
#         plot_name=f"pca_test_projected_pc1_vs_pc2_{name_part}",
#         save_plot=True,
#         custom_title_prefix=custom_prefix
#     )

#     # Test with severity coloring (PC1 vs PC2)
#     clustering_test.transform_test_with_pca(
#         pca=pca_model,
#         x_test=spectra_test,
#         y_meta_test=meta_test.copy(),
#         preprocessing=methods,
#         pc_axes=(1, 2),
#         plot_name=f"pca_test_projected_pc1_vs_pc2_severity_{name_part}",
#         save_plot=True,
#         custom_title_prefix=custom_prefix + " - Severity"
#     )

#     # Test colored by ADF
#     clustering_test.transform_test_with_pca(
#         pca=pca_model,
#         x_test=spectra_test,
#         y_meta_test=meta_test.copy(),
#         preprocessing=methods,
#         pc_axes=(1, 2),
#         plot_name=f"pca_test_projected_pc1_vs_pc2_ADF_{name_part}",
#         save_plot=True,
#         custom_title_prefix=custom_prefix + " - ADF",
#         target_column='ADF_pp_avg'
#     )

#     # Test colored by HGS
#     clustering_test.transform_test_with_pca(
#         pca=pca_model,
#         x_test=spectra_test,
#         y_meta_test=meta_test.copy(),
#         preprocessing=methods,
#         pc_axes=(1, 2),
#         plot_name=f"pca_test_projected_pc1_vs_pc2_HGS_{name_part}",
#         save_plot=True,
#         custom_title_prefix=custom_prefix + " - HGS",
#         target_column='HGS_pp_avg'
#     )




# # ───────────── PCA ADF & HGS for Train and Test (All Preprocessing Combos) ─────────────

# for methods in preprocessing_combos:
#     name_part = '__'.join(methods) if methods else 'no_preprocessing'

#     # ADF - Train
#     clustering_train.perform_pca_and_plot_with_target(
#         x_array=spectra_train,
#         target_column='ADF_pp_avg',
#         plot_name=f"pca_train_only_pc1_vs_pc2_ADF_{name_part}",
#         pc_axes=(1, 2),
#         save_plot=True,
#         preprocessing=methods,
#         custom_title_prefix="Train Set"
#     )

#     # ADF - Test
#     clustering_test.perform_pca_and_plot_with_target(
#         x_array=spectra_test,
#         target_column='ADF_pp_avg',
#         plot_name=f"pca_test_only_pc1_vs_pc2_ADF_{name_part}",
#         pc_axes=(1, 2),
#         save_plot=True,
#         preprocessing=methods,
#         custom_title_prefix="Test Set"
#     )

#     # HGS - Train
#     clustering_train.perform_pca_and_plot_with_target(
#         x_array=spectra_train,
#         target_column='HGS_pp_avg',
#         plot_name=f"pca_train_only_pc1_vs_pc2_HGS_{name_part}",
#         pc_axes=(1, 2),
#         save_plot=True,
#         preprocessing=methods,
#         custom_title_prefix="Train Set"
#     )

#     # HGS - Test
#     clustering_test.perform_pca_and_plot_with_target(
#         x_array=spectra_test,
#         target_column='HGS_pp_avg',
#         plot_name=f"pca_test_only_pc1_vs_pc2_HGS_{name_part}",
#         pc_axes=(1, 2),
#         save_plot=True,
#         preprocessing=methods,
#         custom_title_prefix="Test Set"
#     )


# #%% Clustering Analysis

# # Define the regions (first region, second region, whole spectra)
# regions = {
#     'First Region': all_spectra[:968, :],
#     'Second Region': all_spectra[968:, :],
#     'Whole Spectra': all_spectra
# }

# # Run analysis for all regions and preprocessing methods
# results_clustering_df = run_analysis(all_spectra, y_label_df, regions)

# #%% Signal-To-Noise Ratio for Comparing Substrate Evaluation Techniques

# # Perform substrate evaluation and get the peak heights as a DataFrame
# peak_heights_df = plot_spectra_with_baseline(
#     file_path=method_data_directory,
#     output_dir=output_dir,
#     save_as_png=True,  # Save the plot as a PNG
#     png_name='baseline_peak_height_plot.png',  # Name of the saved plot
#     title_fontsize=36,
#     axis_fontsize=30,
#     label_fontsize=20,
#     peak_label_fontsize=18,
#     line_width=3
# )

# # Convert DataFrame to dictionary with three keys, each containing a 9x3 numpy array
# peak_heights_dict = {
#     'Treated': np.array(peak_heights_df['Treated'].tolist()),
#     'Untreated': np.array(peak_heights_df['Untreated'].tolist()),
#     'No_Ag': np.array(peak_heights_df['No_Ag'].tolist())
# }

# #%% Classification Models


# # Initialize the classifier
# classifier_model = Classifier(all_spectra=all_spectra, y_label_df=y_label_df)

# # Run PLS-DA with LOSOCV and save plots to the output directory
# roc_auc_plsda, all_y_test_plsda, all_y_pred_proba_plsda = classifier_model.pls_da_losocv(
#     preprocess_methods=['SNV', 'Normalization'], 
#     n_components=15, 
#     save_plots=True, 
#     output_dir=output_dir
# )

# # Perform SVM with LOSOCV and save plots to the output directory
# roc_auc_svm, all_y_test_svm, all_y_pred_proba_svm = classifier_model.svm_losocv(
#     preprocess_methods=['SNV'], 
#     C=1.0, 
#     kernel='linear', 
#     save_plots=True, 
#     output_dir=output_dir
# )


#%% Regression Models


############################ PLS ############################
 
# Initialize the regression model
# regression_model = RegressionModel(all_spectra=all_spectra, y_label_df= y_label_df, wavenumbers=wavenumbers)


# pls_results = regression_model.run_pls_hyperparameter_optimization(output_dir=output_dir)


# # Plot the regression results without any preprocessing and with 12 components
# y_test_pls1, y_pred_pls1, stats_pls1 = regression_model.plot_regression_results_pls(
#     preprocess_methods=[],  # No preprocessing
#     n_components=12, 
#     plot_number=1,
#     save_plot=True,  
#     output_dir=output_dir,  
#     axis_fontsize=20, 
#     legend_fontsize=14, 
#     title_fontsize=20
# )


# # Plot the regression results with Normalization preprocessing and 11 components
# y_test_pls2, y_pred_pls2, stats_pls2 =regression_model.plot_regression_results_pls(
#     preprocess_methods=['Normalization'],  # Apply Normalization preprocessing
#     n_components=11, 
#     plot_number=2,
#     save_plot=True,  
#     output_dir=output_dir,  
#     axis_fontsize=20, 
#     legend_fontsize=14, 
#     title_fontsize=20
# )

# # Plot the regression results with Normalization and SNV preprocessing and 3 components
# y_test_pls3, y_pred_pls3, stats_pls3 =regression_model.plot_regression_results_pls(
#     preprocess_methods=['Normalization', 'SNV'],  # Apply Normalization and SNV preprocessing
#     n_components=3, 
#     plot_number=3,
#     save_plot=True,  
#     output_dir=output_dir,  
#     axis_fontsize=16, 
#     legend_fontsize=14, 
#     title_fontsize=20
# )



# ############################ RF ############################

# # Initialize the regression model
# regression_model = RegressionModel(all_spectra=all_spectra, y_label_df=y_label_df)

# # Run Random Forest hyperparameter optimization
# rf_results = regression_model.run_rf_hyperparameter_optimization(output_dir=output_dir, num_iterations=10)


# y_test_rf1, y_pred_rf1, stats_rf1 = regression_model.plot_rf_results(
#     preprocess_methods=['Normalization'],  # Apply Normalization preprocessing
#     rf_params={'n_estimators': 50, 'max_features': 'sqrt', 'max_depth': 25, 'min_samples_split': 180, 'min_samples_leaf': 48}, 
#     plot_number=1,
#     save_plot=True,  
#     output_dir=output_dir,  
#     axis_fontsize=20, 
#     legend_fontsize=14, 
#     title_fontsize=20
# )


# y_test_rf2, y_pred_rf2, stats_rf2 = regression_model.plot_rf_results(
#     preprocess_methods=['SNV'],  # Apply Normalization preprocessing
#     rf_params={'n_estimators': 190, 'max_features': 'sqrt', 'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 40}, 
#     plot_number=2,
#     save_plot=True,  
#     output_dir=output_dir,  
#     axis_fontsize=20, 
#     legend_fontsize=14, 
#     title_fontsize=20
# )


# y_test_rf3, y_pred_rf3, stats_rf3 = regression_model.plot_rf_results(
#     preprocess_methods=['SNV'],  # Apply Normalization preprocessing
#     rf_params={'n_estimators': 50, 'max_features': 'sqrt', 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 55}, 
#     plot_number=3,
#     save_plot=True,  
#     output_dir=output_dir,  
#     axis_fontsize=16, 
#     legend_fontsize=14, 
#     title_fontsize=20
# )


############################ iPLS ############################

# Initialize the regression model
# regression_model = RegressionModel(all_spectra=all_spectra, y_label_df=y_label_df, wavenumbers=wavenumbers)



# # Run iPLS hyperparameter optimization
# ipls_results = regression_model.run_ipls_hyperparameter_optimization(
#     output_dir=output_dir,
#     fold_log_dir=fold_log_dir
# )

# # Save iPLS R² results to Excel
# excel_output_dir = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\Model R2 Results Excel"
# os.makedirs(excel_output_dir, exist_ok=True)

# ipls_results_file = os.path.join(excel_output_dir, "iPLS_Hyperparameter_R2_Results.xlsx")
# ipls_results.to_excel(ipls_results_file, index=False)

# # Automatically generate parity plots for the best model per target
# target_mapping = {
#     'target_SI': 1,
#     'HGS_pp_avg': 2,
#     'ADF_pp_avg': 3
# }

# for metric, plot_number in target_mapping.items():
#     best_row = ipls_results.loc[ipls_results[f'Sample-Level R² {metric}'].idxmax()]
#     preprocess_methods = best_row['Preprocessing Methods'].split(' + ') if best_row['Preprocessing Methods'] else []
#     n_components = int(best_row['N Components'])
#     num_intervals = int(best_row['Num Intervals'])

#     regression_model.plot_ipls_results(
#         preprocess_methods=preprocess_methods,
#         n_components=n_components,
#         num_intervals=num_intervals,
#         plot_number=plot_number,
#         save_plot=True,
#         output_dir=output_dir,
#         axis_fontsize=20,
#         legend_fontsize=14,
#         title_fontsize=20
#     )
# # Save inner folds for the best models per target
# inner_fold_log_dir = os.path.join(fold_log_dir, "Best_Model_Inner_Folds")
# os.makedirs(inner_fold_log_dir, exist_ok=True)

# regression_model.log_inner_folds_for_best_models(
#     optimization_df=ipls_results,
#     base_dir=inner_fold_log_dir
# )
# # ================= Evaluate Best iPLS Models on External Test Set =================

# # Define paths to external test set
# external_test_spectra_path = r"C:\Users\notha\Downloads\Data\data_test_updated"
# external_test_metadata_path = r"C:\Users\notha\Downloads\Data\y_metadata_test_updated.csv"
# external_test_output_dir = r"C:\Users\notha\Downloads\Data\Results"

# os.makedirs(external_test_output_dir, exist_ok=True)
# # Load test spectra for overlay plot
# _, _, test_all_spectra, test_filenames = load_data(external_test_spectra_path, return_filenames=True)
# # Merge filenames for MCCV use
# # Sanity check before concatenating
# print(f"Train filenames: {len(filenames_per_column)}")
# print(f"Test filenames: {len(test_filenames)}")

# combined_filenames = filenames_per_column + test_filenames
# print(f"Combined filenames: {len(combined_filenames)}")  # Should equal 441 + 144 = 585

# X_all_raw = np.hstack([all_spectra, test_all_spectra])

# print(f"X_all_raw shape: {X_all_raw.shape}")  # Should be (1731, 585)

# # Confirm consistency
# assert X_all_raw.shape[1] == len(combined_filenames), "Mismatch between filenames and spectra!"

# # Pass correct filenames
# from modules.mccv_outlier_detection import run_mccv_outlier_detection
# run_mccv_outlier_detection(
#     train_data_dir=data_directory,
#     train_meta_path=meta_data_directory,
#     test_data_dir=external_test_spectra_path,
#     test_meta_path=external_test_metadata_path,
#     filenames_per_column=combined_filenames  # ✅ use correct merged list
# )


# # Plot training vs test spectra overlay
# plot_train_test_spectra(
#     wavenumbers=wavenumbers,
#     train_spectra=all_spectra,
#     test_spectra=test_all_spectra,
#     train_filenames=filenames_per_column,
#     test_filenames=test_filenames,
#     output_path=os.path.join(output_dir, 'train_vs_test_spectra_overlay.png')
# )


# # Run prediction and save plots
# regression_model.predict_on_test_set_with_best_ipls_models(
#     optimization_df=ipls_results,
#     test_spectra_path=external_test_spectra_path,
#     test_metadata_path=external_test_metadata_path,
#     output_dir=external_test_output_dir
# )
# # ================= Evaluate Ensemble iPLS Model on External Test Set =================

# ensemble_output_dir = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\Ensemble_excel_and_parity"
# os.makedirs(ensemble_output_dir, exist_ok=True)

# # Choose your best settings manually or extract from ipls_results
# regression_model.predict_test_set_with_ipls_ensemble(
#     test_spectra_path=external_test_spectra_path,
#     test_metadata_path=external_test_metadata_path,
#     preprocess_methods=['Normalization', 'SNV'],  # <-- use same as best from ipls_results if needed
#     n_components=5,                               # <-- choose based on top model
#     num_intervals=15,                             # <-- choose based on top model
#     n_splits=10,
#     output_dir=ensemble_output_dir
# )

# ############################ EJCR ############################

# run_ejcr_analysis(
#     y_test_ipls1, 
#     y_pred_ipls1, 
#     title="Splicing Index", 
#     save=True, 
#     output_filename=os.path.join(output_dir, "SI_EJCR.png")
# )


# run_ejcr_analysis(
#     y_test_ipls2, 
#     y_pred_ipls2, 
#     title="Hand Grip Strength (%)", 
#     save=True, 
#     output_filename=os.path.join(output_dir, "HGS_EJCR.png")
# )


# run_ejcr_analysis(
#     y_test_rf3, 
#     y_pred_rf3, 
#     title="Average Ankle Dorsiflexion(%)", 
#     save=True, 
#     output_filename=os.path.join(output_dir, "ADF_EJCR.png")
# )



# # ############################ VIP-Scores ############################

# # # Create the first VIP plot
# create_vip_plot(
#     all_spectra=all_spectra, 
#     y_label_df=y_label_df, 
#     wavenumber=wavenumbers, 
#     output_dir=output_dir, 
#     file_name='vip_plot_emsc_second_derivative', 
#     preprocess_methods=['EMSC', 'Second Derivative'], 
#     n_components=12
# )

# # Create the second VIP plot
# create_vip_plot(
#     all_spectra=all_spectra, 
#     y_label_df=y_label_df, 
#     wavenumber=wavenumbers, 
#     output_dir=output_dir, 
#     file_name='vip_plot_normalization_snv', 
#     preprocess_methods=['Normalization', 'SNV'], 
#     n_components=3
# )

# ================= SiPLS Pipeline =================

# # Define SiPLS output directory
# sipls_output_dir = r"C:\Users\notha\Downloads\Data\Results"
# os.makedirs(sipls_output_dir, exist_ok=True)

# # Run SiPLS hyperparameter optimization
# sipls_results = regression_model.run_sipls_hyperparameter_optimization(
#     output_dir=sipls_output_dir,
#     fold_log_dir=os.path.join(sipls_output_dir, "Fold Logs")
# )

# Save results to Excel (already done inside the function, so this is optional)
# sipls_results.to_excel(os.path.join(sipls_output_dir, "SiPLS_Hyperparameter_Results.xlsx"), index=False)

# # Automatically plot best models per target
# target_mapping = {
#     'target_SI': 1,
#     'HGS_pp_avg': 2,
#     'ADF_pp_avg': 3
# }

# for metric, plot_number in target_mapping.items():
#     best_row = sipls_results.loc[sipls_results[f'Sample-Level R² {metric}'].idxmax()]
#     preprocess_methods = best_row['Preprocessing Methods'].split(' + ') if best_row['Preprocessing Methods'] else []
#     n_components = int(best_row['N Components'])
#     num_intervals = int(best_row['Num Intervals'])
#     max_combo_size = int(best_row['Max Combo Size'])

#     regression_model.plot_sipls_results(
#         preprocess_methods=preprocess_methods,
#         n_components=n_components,
#         num_intervals=num_intervals,
#         max_combination_size=max_combo_size,
#         plot_number=plot_number,
#         save_plot=True,
#         output_dir=sipls_output_dir,
#         axis_fontsize=20,
#         legend_fontsize=14,
#         title_fontsize=20
#     )

# # ================= SiPLS Test Set Ensemble =================

# sipls_ensemble_output_dir = os.path.join(sipls_output_dir, "Ensemble Test Set")
# os.makedirs(sipls_ensemble_output_dir, exist_ok=True)

# # 🔽 Extract best SiPLS model settings dynamically from results
# best_row = sipls_results.loc[sipls_results['Sample-Level R² target_SI'].idxmax()]
# preprocess_methods = best_row['Preprocessing Methods'].split(' + ') if best_row['Preprocessing Methods'] else []
# n_components = int(best_row['N Components'])
# num_intervals = int(best_row['Num Intervals'])
# max_combo_size = int(best_row['Max Combo Size'])

# # 🔁 Now use these values instead of hardcoding
# regression_model.predict_test_set_with_sipls_ensemble(
#     test_spectra_path=external_test_spectra_path,
#     test_metadata_path=external_test_metadata_path,
#     preprocess_methods=preprocess_methods,
#     n_components=n_components,
#     num_intervals=num_intervals,
#     max_combination_size=max_combo_size,
#     n_splits=10,
#     output_dir=sipls_ensemble_output_dir
# )

# # ================= Plot Final Selected SiPLS Intervals =================

# # Extract best intervals from the best-performing target
# best_metric = 'Sample-Level R² target_SI'
# best_row = sipls_results.loc[sipls_results[best_metric].idxmax()]
# best_intervals = best_row['Best Interval Combo']

# # This will be a list of (start_cm, end_cm) tuples already
# regression_model.plot_selected_sipls_intervals(
#     selected_intervals=best_intervals,
#     output_dir=sipls_output_dir,
#     title="Best SiPLS Intervals for Splicing Index",
#     save_plot=True,
#     show_plot=True,
#     plot_average=True  # Or False to see individual spectra
# )
# # ================= SiPLS SPXY Evaluation =================
# spxy_output_dir = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SiPLS SPXY\SPXY RESULTS"
# os.makedirs(spxy_output_dir, exist_ok=True)

# # List of preprocessing combinations to iterate through
# preprocessing_combinations = [
#     [],  # No preprocessing
#     # ['Normalization'],
#     # ['SNV'],
#     # ['Normalization', 'SNV'],
#     # ['EMSC'],
#     # ['Second Derivative'],
#     # ['EMSC', 'Second Derivative'],
#     # ['Normalization', 'Second Derivative'],
#     # ['SNV', 'Second Derivative']
# ]

# # Loop through and evaluate each combination
# for methods in preprocessing_combinations:
#     safe_name = '__'.join(methods) if methods else 'no_preprocessing'
#     combo_output_dir = os.path.join(spxy_output_dir, safe_name)
#     os.makedirs(combo_output_dir, exist_ok=True)

#     regression_model.evaluate_test_set_with_sipls_spxy(
#         test_spectra_path=external_test_spectra_path,
#         test_metadata_path=external_test_metadata_path,
#         preprocess_methods=methods,
#         n_components_list=[2 ],
#         num_intervals_list=[10],
#         max_combination_size=4,
#         output_dir=combo_output_dir
#     )
# ────────────────────────────────────────────────────────────────
# ── MCCV OUTLIER DETECTION
# ────────────────────────────────────────────────────────────────

# from modules.mccv_outlier_detection import run_mccv_outlier_detection

# run_mccv_outlier_detection(
#     train_data_dir=data_directory,
#     train_meta_path=meta_data_directory,
#     test_data_dir=external_test_spectra_path,
#     test_meta_path=external_test_metadata_path,
#     filenames_per_column=filenames_per_column,  # <-- add this
# )


# ────────────────────────────────────────────────────────────────
# ── Mapping Analysis
# ────────────────────────────────────────────────────────────────
# from modules.mapping_analysis import run_mapping_comparison

# run_mapping_comparison(
#     data_dir=data_directory,
#     metadata_path=meta_data_directory,
#     output_dir=r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Mappings_Comparison"
# )
# # ────────────────────────────────────────────────────────────────
# ── Domain Shift Visualization: Compare Train vs Test Targets ──
# ────────────────────────────────────────────────────────────────
# from modules.domain_shift_plot import plot_domain_shift

# # Define metadata paths and output directory
# train_meta_path = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\y_metadata.csv"
# test_meta_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata_test_updated.csv"
# output_dir = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\figures_original"

# # # Run domain shift visualization
# # plot_domain_shift(train_meta_path, test_meta_path, output_dir)

# # from modules.plot_spectra import plot_train_and_test_separately

# # # Corrected call:
# # plot_train_and_test_separately(
# #     wavenumbers=wavenumbers,
# #     train_spectra=all_spectra,
# #     test_spectra=test_all_spectra,
# #     train_filenames=filenames_per_column,
# #     test_filenames=test_filenames,
# #     output_dir=output_dir
# # )
# # ---------------------------CREATES 3d PLOT FOF TRAINING SET ----------------------
# # Set path to your training metadata
# train_meta_path = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\y_metadata.csv"
# output_3d_path = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\figures_original\3d_plot_si_adf_hgs.png"

# # Call the function
# plot_3d_target_metrics(train_meta_path, output_3d_path)

# ──────────────────────────────────────────────────────────────────────
# 4) SPXY‐SiPLS hyperparameter optimization + expanded intervals
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────
# SPXY-SiPLS with the full Preprocessing API
# ──────────────────────────────────────────────

# # Paths to your blind test set
# external_test_spectra_path  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data_test_updated"
# # old
# # external_test_metadata_path = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\y_metadata_test.csv"
# # New 11 sample updated test set
# external_test_metadata_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata_test_updated.csv"

# # Make an output folder for SPXY-SiPLS
# sipls_spxy_out = os.path.join(output_dir, "SiPLS_SPXY_Results")
# os.makedirs(sipls_spxy_out, exist_ok=True)

# # Initialize the SPXY‐SiPLS model
# reg_model = RegressionModel(
#     all_spectra=all_spectra,
#     y_label_df=y_label_df,
#     wavenumbers=wavenumbers
# )

# # Run SPXY‐SiPLS over SI ONLY:
# #   • all 5 pretreatments (none, EMSC, Normalization, SNV, 2nd Derivative)
# #   • C = 2,3,4,5,7,8,9
# #   • I = 10,15,20,…,150
# #   • ±50 shift on the final best combo
# reg_model.evaluate_test_set_with_sipls_spxy(
#     test_spectra_path=external_test_spectra_path,
#     test_metadata_path=external_test_metadata_path,
#     # preprocess_methods_list is None + uses all 5 by default
#     n_components_list=[5,6],
#     num_intervals_list=[10],
#     max_shift=30,
#     max_combo_intervals=4,
#     output_dir=sipls_spxy_out
# )
# Final SPXY-SiPLS Evaluation on SI, HGS, and ADF
#### THIS IS THE LASTEST WORKING SIPLS
# Each run uses all 5 preprocessing methods (None, EMSC, SNV, Normalization, 2nd Derivative)
# for idx, name in zip([0, 1, 2], ["SI", "HGS", "ADF"]):
#     reg_model.evaluate_test_set_with_sipls_spxy(
#         test_spectra_path=external_test_spectra_path,
#         test_metadata_path=external_test_metadata_path,
#         target_index=idx,
#         target_name=name,
#         n_components_list=[2,3,4,5,6,7,8,9],
#         num_intervals_list=[1,10],
#         max_shift=20,
#         max_combo_intervals=4,
#         output_dir=sipls_spxy_out
#     )
# ──────────────────────────────────────────────
# SPXY MULTI MODEL INTERVAL SELECTION
# ──────────────────────────────────────────────
# Example run for target SI using RF, XGB, SVR, etc.
# run_interval_selection_and_modeling(
#     all_spectra=all_spectra,
#     y_label_df=y_label_df,
#     wavenumbers=wavenumbers,
#     target_name='SI',  # or 'ADF' or 'HGS'
#     test_spectra_path=external_test_spectra_path,
#     test_metadata_path=external_test_metadata_path,
#     output_dir=os.path.join(output_dir, "MultiModel_Interval_Selection_Results"),
#     model_list=[
#         'sipls',         # TorchPLS (SiPLS with GPU support)
#         # 'random_forest', # sklearn.ensemble.RandomForestRegressor
#         # 'xgboost',       # xgboost.XGBRegressor
#         # 'svr',           # sklearn.svm.SVR (RBF kernel)
#         # 'mlp',           # sklearn.neural_network.MLPRegressor
#         # 'knn',           # sklearn.neighbors.KNeighborsRegressor
#         # 'gpr'            # sklearn.gaussian_process.GaussianProcessRegressor
#     ]
# )
# =============================================================================
from itertools import product

# Xg-Boost Param Grid
# =============================================================================
# 1) Base (best n_estimators / lr ranges)
base = {
    'n_estimators':  [100, 200],
    'learning_rate': [0.05, 0.1],
}

# 2) Tree complexity
tree = {
    'max_depth':        [3, 5, 7],
    'min_child_weight': [1, 3, 5],
}

# 3) Subsampling & feature-sampling
split = {
    'gamma':            [0, 0.1, 0.2],
    'subsample':        [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

# 4) Optional L1/L2
reg = {
    'reg_alpha':  [0, 0.1, 1.0],
    'reg_lambda': [1.0, 1.5, 2.0],
}

# Build the full XGB grid
xgb_grid = []
for ne, lr, md, mcw, ga, ss, cs, ra, rl in product(
        base['n_estimators'], base['learning_rate'],
        tree['max_depth'], tree['min_child_weight'],
        split['gamma'], split['subsample'], split['colsample_bytree'],
        reg['reg_alpha'], reg['reg_lambda']
    ):
    xgb_grid.append({
        'n_estimators':     ne,
        'learning_rate':    lr,
        'max_depth':        md,
        'min_child_weight': mcw,
        'gamma':            ga,
        'subsample':        ss,
        'colsample_bytree': cs,
        'reg_alpha':        ra,
        'reg_lambda':       rl,
        'n_jobs':          -1,   # will be overwritten below
    })

# Use GPU and single-thread CPU for each trial
for params in xgb_grid:
    params.update({
        'n_jobs':       1,
        'tree_method':  'gpu_hist',
        'predictor':    'gpu_predictor',
    })

print("🔍 xgb_grid size:", len(xgb_grid))
print("First 2 entries:", xgb_grid[:2])


# 10-Fold Leave-One-Out Blinded Test Evaluation
# =============================================================================

from modules.ten_fold_loo_shell_evaluation import leave_one_out_test_evaluation
from modules.interval_shell import DEFAULT_HYPERPARAMETERS

# ── Models to try (include or comment out as you like) ───────────────────────
model_list = [
    # 'sipls',
    # 'random_forest',
    # 'svr',
    'xgboost',
    # 'mlp',
    # 'knn',
    # 'gpr'
]

# =============================================================================
# 10-Fold LOO hyperparameter grids (add or comment entries as needed)
# =============================================================================
hyperparam_grids = {
    'sipls': [
        {'n_components': 2, 'device': 'cpu'},
        {'n_components': 5, 'device': 'cpu'},
        {'n_components': 5, 'device': 'cuda'},
        {'n_components': 8, 'device': 'cpu'},
        {'n_components': 10, 'device': 'cuda'},
    ],
    'random_forest': [
        {'n_estimators': 50, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': None},
        {'n_estimators': 200, 'max_depth': 20},
    ],
    'xgboost': xgb_grid,
    #     [
    #     {'n_estimators': 100, 'learning_rate': 0.1},
    #     {'n_estimators': 200, 'learning_rate': 0.05},
    #     {'n_estimators': 300, 'learning_rate': 0.01},
    # ],
    'svr': [
        {'C': 1.0, 'epsilon': 0.1},
        {'C': 10.0, 'epsilon': 0.1},
        {'C': 10.0, 'epsilon': 0.2},
    ],
    'mlp': [
        {'hidden_layer_sizes': (100,), 'max_iter': 1000},
        {'hidden_layer_sizes': (200,), 'max_iter': 1000},
        {'hidden_layer_sizes': (100,100), 'max_iter': 1000},
    ],
    'knn': [
        {'n_neighbors': 3},
        {'n_neighbors': 5},
        {'n_neighbors': 7},
    ],
    'gpr': [
        {'alpha': 1e-10},
        {'alpha': 1e-5},
        {'alpha': 1e-2},
    ],
}
param_grid = {
    "learning_rate": [0.001, 0.01, 0.1],
    "num_hidden_layers": [2, 3, 4],
    "num_neurons_per_layer": [32, 64, 128],
    "activation_function": ["relu", "tanh", "sigmoid"],
    "batch_size": [16, 32, 64],
    "optimizer": ["adam", "sgd"],
    "dropout_rate": [0.0, 0.2, 0.5]
}

# ── Preprocessing chains ────────────────────────────────────────────────────
preprocess_grid = [
    [],                    # no preprocessing
    # ['EMSC'],
    # ['IntervalPLS'],
    # ['SNV'],
    # ['SNV', 'IntervalPLS'],
    # ['Normalization', 'IntervalPLS'],
    # ['IntervalPLS', 'Normalization' ],
    # ['Second Derivative'],
    # ['EMSC', 'SNV'],
    # ['IntervalPLS','EMSC', 'SNV'],
    # ['EMSC', 'SNV', 'IntervalPLS' ],
    # ['SNV', 'Second Derivative']
]

# =============================================================================
# Targets
# =============================================================================
targets = [
    ("Splicing Index",     "target_SI"),
    # ("Hand‐Grip Strength", "HGS_pp_avg"),
    # ("Ankle Dorsiflexion",  "ADF_pp_avg")
]

# ── Set your new data locations here ────────────────────────────────────────
data_directory              = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
meta_data_directory         = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"
external_test_spectra_path  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data_test_updated"
external_test_metadata_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata_test_updated.csv"
output_dir                  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\10_fold_results"

# =============================================================================
# Run the Leave-One-Out evaluator for each target
# =============================================================================
for nice_name, col in targets:
    out_dir_col = os.path.join(output_dir, f'10_fold_LOO_{col}')
    print(f"\n\n### Running LOO for {nice_name} ({col}) – saving to: {out_dir_col}")
    loo_df, loo_metrics = leave_one_out_test_evaluation(
        train_data_dir   = data_directory,
        train_meta_path  = meta_data_directory,
        test_data_dir    = external_test_spectra_path,
        test_meta_path   = external_test_metadata_path,
        model_list       = model_list,
        hyperparam_grids = hyperparam_grids,
        preprocess_grid  = preprocess_grid,
        output_dir       = out_dir_col,
        target_column    = col
    )

    print(f"--- {nice_name} predictions head ---")
    print(loo_df.head())
    print(f"--- {nice_name} aggregated metrics ---")
    print(loo_metrics)

