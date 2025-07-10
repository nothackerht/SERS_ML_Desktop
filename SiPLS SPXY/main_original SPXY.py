
"""

@author: Matthew Glace (glacemk@vcu.edu)

Virginia Commonwealth University, 2024

Data Analysis for Surface-Enhanced Raman Spectroscopy of Muscular Dystrophy Plasma Samples

"""


#%% Data Loading 

from modules.data_loader import load_data, load_metadata
from modules.unsupervised_clustering import UnsupervisedClustering
from modules.plot_spectra import create_plot
from modules.analysis_tools import run_analysis, run_ejcr_analysis, create_vip_plot
from modules.substrate_evaluation import plot_spectra_with_baseline
from modules.classification import Classifier
from modules.regression_original import RegressionModel
from modules.plot_spectra import plot_train_test_spectra


import numpy as np
import os

# Specify the directory containing the Data and Metadata
data_directory = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\data"
meta_data_directory = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\y_metadata.csv"
method_data_directory = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\method_data.csv"

output_dir = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\figures_original"

fold_log_dir = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\Fold DOcumentation"


# Load the Data and Metadata
wavenumbers, averaged_spectra, all_spectra = load_data(data_directory)
y_label_df = load_metadata(meta_data_directory)

#%% Plot Averaged Spectral Data

# Plot and save the spectra
create_plot(wavenumbers, all_spectra, y_label_df, f'{output_dir}/all_spectra_plot_combined.png')

# #%% Principal Component Analysis 

# # Initialize the UnsupervisedClustering class
# clustering = UnsupervisedClustering(y_label_df, all_spectra, output_dir)

# # Perform and plot PCA for PC1 vs PC2 with no preprocessing
# clustering.perform_pca_and_plot(
#     x_array=clustering.spectra,
#     plot_name='pca_pc1_vs_pc2_no_preprocessing',
#     pc_axes=(1, 2),
#     save_plot=True
# )

# # Perform and plot PCA for PC1 vs PC3 with no preprocessing
# clustering.perform_pca_and_plot(
#     x_array=clustering.spectra,
#     plot_name='pca_pc1_vs_pc3_no_preprocessing',
#     pc_axes=(1, 3),
#     save_plot=True
# )

# # Perform and plot PCA with severity for PC1 vs PC2
# clustering.perform_pca_and_plot_with_severity(
#     x_array=clustering.spectra,
#     plot_name='pca_severity_pc1_vs_pc2',
#     pc_axes=(1, 2),
#     save_plot=True
# )

# # Perform and plot PCA with severity for PC1 vs PC3
# clustering.perform_pca_and_plot_with_severity(
#     x_array=clustering.spectra,
#     plot_name='pca_severity_pc1_vs_pc3',
#     pc_axes=(1, 3),
#     save_plot=True
# )

# # Perform and plot PCA for PC1 vs PC2 with EMSC preprocessing
# clustering.perform_pca_and_plot(
#     x_array=clustering.spectra,
#     plot_name='pca_pc1_vs_pc2_emsc',
#     pc_axes=(1, 2),
#     save_plot=True,
#     preprocessing=['EMSC']
# )

# # Perform and plot PCA for PC1 vs PC2 with EMSC preprocessing and specific axis bounds
# clustering.perform_pca_and_plot(
#     x_array=clustering.spectra,
#     plot_name='pca_pc1_vs_pc2_emsc_with_bounds',
#     pc_axes=(1, 2),
#     save_plot=True,
#     preprocessing=['EMSC'],
#     x_bounds=(-0.2e6, 0.2e6),  # Custom x-axis bounds
#     y_bounds=(-0.2e5, 0.2e5)   # Custom y-axis bounds
# )

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
regression_model = RegressionModel(all_spectra=all_spectra, y_label_df= y_label_df, wavenumbers=wavenumbers)


pls_results = regression_model.run_pls_hyperparameter_optimization(output_dir=output_dir)


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
regression_model = RegressionModel(all_spectra=all_spectra, y_label_df=y_label_df, wavenumbers=wavenumbers)



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

# Define paths to external test set
external_test_spectra_path = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\data_test"
external_test_metadata_path = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\y_metadata_test.csv"
external_test_output_dir = r"C:\Users\spect\Downloads\MD-Analysis-main (3)\MD-Analysis-main_OG\Test Set Results"

os.makedirs(external_test_output_dir, exist_ok=True)
# Load test spectra for overlay plot
_, _, test_all_spectra = load_data(external_test_spectra_path)

# Plot training vs test spectra overlay
plot_train_test_spectra(
    wavenumbers=wavenumbers,
    train_spectra=all_spectra,
    test_spectra=test_all_spectra,
    output_path=os.path.join(output_dir, 'train_vs_test_spectra_overlay.png')
)


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

# Define SiPLS output directory
sipls_output_dir = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SiPLS APPROACH"
os.makedirs(sipls_output_dir, exist_ok=True)

# Run SiPLS hyperparameter optimization
sipls_results = regression_model.run_sipls_hyperparameter_optimization(
    output_dir=sipls_output_dir,
    fold_log_dir=os.path.join(sipls_output_dir, "Fold Logs")
)

# Save results to Excel (already done inside the function, so this is optional)
# sipls_results.to_excel(os.path.join(sipls_output_dir, "SiPLS_Hyperparameter_Results.xlsx"), index=False)

# Automatically plot best models per target
target_mapping = {
    'target_SI': 1,
    'HGS_pp_avg': 2,
    'ADF_pp_avg': 3
}

for metric, plot_number in target_mapping.items():
    best_row = sipls_results.loc[sipls_results[f'Sample-Level R² {metric}'].idxmax()]
    preprocess_methods = best_row['Preprocessing Methods'].split(' + ') if best_row['Preprocessing Methods'] else []
    n_components = int(best_row['N Components'])
    num_intervals = int(best_row['Num Intervals'])
    max_combo_size = int(best_row['Max Combo Size'])

    regression_model.plot_sipls_results(
        preprocess_methods=preprocess_methods,
        n_components=n_components,
        num_intervals=num_intervals,
        max_combination_size=max_combo_size,
        plot_number=plot_number,
        save_plot=True,
        output_dir=sipls_output_dir,
        axis_fontsize=20,
        legend_fontsize=14,
        title_fontsize=20
    )

# ================= SiPLS Test Set Ensemble =================

sipls_ensemble_output_dir = os.path.join(sipls_output_dir, "Ensemble Test Set")
os.makedirs(sipls_ensemble_output_dir, exist_ok=True)

# 🔽 Extract best SiPLS model settings dynamically from results
best_row = sipls_results.loc[sipls_results['Sample-Level R² target_SI'].idxmax()]
preprocess_methods = best_row['Preprocessing Methods'].split(' + ') if best_row['Preprocessing Methods'] else []
n_components = int(best_row['N Components'])
num_intervals = int(best_row['Num Intervals'])
max_combo_size = int(best_row['Max Combo Size'])

# 🔁 Now use these values instead of hardcoding
regression_model.predict_test_set_with_sipls_ensemble(
    test_spectra_path=external_test_spectra_path,
    test_metadata_path=external_test_metadata_path,
    preprocess_methods=preprocess_methods,
    n_components=n_components,
    num_intervals=num_intervals,
    max_combination_size=max_combo_size,
    n_splits=10,
    output_dir=sipls_ensemble_output_dir
)

# ================= Plot Final Selected SiPLS Intervals =================

# Extract best intervals from the best-performing target
best_metric = 'Sample-Level R² target_SI'
best_row = sipls_results.loc[sipls_results[best_metric].idxmax()]
best_intervals = best_row['Best Interval Combo']

# This will be a list of (start_cm, end_cm) tuples already
regression_model.plot_selected_sipls_intervals(
    selected_intervals=best_intervals,
    output_dir=sipls_output_dir,
    title="Best SiPLS Intervals for Splicing Index",
    save_plot=True,
    show_plot=True,
    plot_average=True  # Or False to see individual spectra
)
