import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from modules.preprocessing import Preprocessing

class Classifier:
    
    def __init__(self, all_spectra, y_label_df):
        self.all_spectra = all_spectra
        self.y_label_df = y_label_df
        self.y_label_df_repeated = pd.DataFrame(np.repeat(y_label_df.values, 9, axis=0), columns=y_label_df.columns)
        self.all_y_pred_proba = None
        self.y_target = None
        self.groups = None
        
    def plot_si_grad(self, font_size=30, save_plot=False, file_path=None):
        si_label = self.y_label_df['target_SI'].to_numpy()[:48]
        ktest_control_pred = []
        ktest_control_sd = []
        ktest_dm1_pred = []
        ktest_dm1_sd = []

        for i in range(0, 6):
            ktest_control_pred.append(np.mean(self.all_y_pred_proba[i * 9:(i + 1) * 9]))
            ktest_control_sd.append(np.std(self.all_y_pred_proba[i * 9:(i + 1) * 9]))

        for i in range(6, 48):
            ktest_dm1_pred.append(np.mean(self.all_y_pred_proba[i * 9:(i + 1) * 9]))
            ktest_dm1_sd.append(np.std(self.all_y_pred_proba[i * 9:(i + 1) * 9]))

        ktest_control_pred = np.array(ktest_control_pred)
        ktest_control_sd = np.array(ktest_control_sd)

        ktest_dm1_pred = np.array(ktest_dm1_pred)
        ktest_dm1_sd = np.array(ktest_dm1_sd)

        sorted_indices = np.argsort(si_label[6:])
        sorted_si_label = si_label[6:][sorted_indices]
        sorted_ktest_dm1_pred = ktest_dm1_pred[sorted_indices]
        sorted_ktest_dm1_sd = ktest_dm1_sd[sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.errorbar(ktest_control_pred, np.zeros_like(ktest_control_pred), xerr=ktest_control_sd, fmt='o', 
                     label='Control', capsize=5, color='blue')
        plt.errorbar(sorted_ktest_dm1_pred, sorted_si_label, xerr=sorted_ktest_dm1_sd, fmt='o', 
                     label='DM1', capsize=5, color='orange')

        plt.ylabel('Splicing Index (SI)', fontsize=font_size, fontname='Arial', fontweight='bold')
        plt.xlabel('Predicted Probability', fontsize=font_size, fontname='Arial', fontweight='bold')
        plt.legend(fontsize=font_size-10)
        plt.xticks(fontsize=font_size, fontname='Arial')
        plt.yticks(fontsize=font_size, fontname='Arial')
        plt.tight_layout()

        if save_plot and file_path:
            plt.savefig(file_path, format='png', dpi=1200)

        plt.show()
         
    def plot_simple(self, font_size=30, save_plot=False, file_path=None):
        unique_samples = np.unique(self.groups)
        mean_preds = []
        true_labels = []

        for sample in unique_samples:
            indices = np.where(self.groups == sample)[0]
            mean_pred = np.mean(self.all_y_pred_proba[indices])
            true_label = self.y_target[sample]

            mean_preds.append(mean_pred)
            true_labels.append(true_label)

        mean_preds = np.array(mean_preds)
        true_labels = np.array(true_labels)

        plt.figure(figsize=(10, 6))
        plt.ylim(-0.5, 1.5)

        control_indices = np.where(true_labels == 0)[0]
        plt.scatter(mean_preds[control_indices], np.zeros_like(control_indices) + 0.1, label='Control', color='blue', s=100, alpha=1, edgecolors='black')

        dm1_indices = np.where(true_labels == 1)[0]
        plt.scatter(mean_preds[dm1_indices], np.ones_like(dm1_indices) - 0.1, label='DM1', color='orange', s=100, alpha=0.75, edgecolors='black')

        plt.yticks([0.1, 0.9], ['Control', 'DM1'], fontsize=font_size, fontname='Arial', fontweight='bold')
        plt.xlabel('Predicted Probability', fontsize=font_size, fontname='Arial', fontweight='bold')
        plt.xticks(fontsize=24, fontname='Arial', fontweight='bold')

        plt.tight_layout()

        if save_plot and file_path:
            plt.savefig(file_path, format='png', dpi=1200)

        plt.show()
        
    def pls_da_losocv(self, preprocess_methods=[], n_components=5, save_plots=False, output_dir=None):
        preprocess = Preprocessing(self.all_spectra)
        x_spectra = preprocess.preprocess(preprocess_methods)
        
        include_indices = self.y_label_df['Type'].isin(['Control', 'DM1'])
        include_indices_all = self.y_label_df_repeated['Type'].isin(['Control', 'DM1'])
        
        self.y_target = self.y_label_df[include_indices]['Type'].reset_index(drop=True).apply(lambda x: 0 if x == 'Control' else 1).values
        self.y_target_repeated = self.y_label_df_repeated[include_indices_all]['Type'].reset_index(drop=True).apply(lambda x: 0 if x == 'Control' else 1).values
        
        self.x_filtered = x_spectra[:, include_indices_all]
        self.groups = np.repeat(np.arange(len(self.y_target)), 9)
        
        logo = LeaveOneGroupOut()
        all_y_test = []
        all_y_pred_proba = []

        X = np.transpose(self.x_filtered)
        y = self.y_target_repeated

        for train_index, test_index in logo.split(X, y, self.groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            pls_da = PLSRegression(n_components=n_components)
            pls_da.fit(X_train, y_train)
            y_pred_proba = pls_da.predict(X_test).ravel()

            all_y_test.append(y_test)
            all_y_pred_proba.append(y_pred_proba)

        self.all_y_test = np.concatenate(all_y_test)
        self.all_y_pred_proba = np.concatenate(all_y_pred_proba)

        fpr, tpr, _ = roc_curve(self.all_y_test, self.all_y_pred_proba)
        roc_auc = roc_auc_score(self.all_y_test, self.all_y_pred_proba)
        
        # Set file paths for saving plots
        if save_plots and output_dir:
            si_plot_path = f'{output_dir}/plsda_si_plot.png'
            simple_plot_path = f'{output_dir}/plsda_simple_plot.png'
        else:
            si_plot_path = None
            simple_plot_path = None

        self.plot_si_grad(save_plot=save_plots, file_path=si_plot_path)
        self.plot_simple(save_plot=save_plots, file_path=simple_plot_path)

        return roc_auc, self.all_y_test, self.all_y_pred_proba

    def svm_losocv(self, preprocess_methods=[], C=1.0, kernel='linear', save_plots=False, output_dir=None):
        preprocess = Preprocessing(self.all_spectra)
        x_spectra = preprocess.preprocess(preprocess_methods)
        
        include_indices = self.y_label_df['Type'].isin(['Control', 'DM1'])
        include_indices_all = self.y_label_df_repeated['Type'].isin(['Control', 'DM1'])
        
        self.y_target = self.y_label_df[include_indices]['Type'].reset_index(drop=True).apply(lambda x: 0 if x == 'Control' else 1).values
        self.y_target_repeated = self.y_label_df_repeated[include_indices_all]['Type'].reset_index(drop=True).apply(lambda x: 0 if x == 'Control' else 1).values
        
        self.x_filtered = x_spectra[:, include_indices_all]
        self.groups = np.repeat(np.arange(len(self.y_target)), 9)
        
        logo = LeaveOneGroupOut()
        all_y_test = []
        all_y_pred_proba = []

        X = np.transpose(self.x_filtered)
        y = self.y_target_repeated

        for train_index, test_index in logo.split(X, y, self.groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            svm_model = SVC(C=C, kernel=kernel, probability=True)
            svm_model.fit(X_train, y_train)
            y_pred_proba = svm_model.predict_proba(X_test)[:, 1]

            all_y_test.append(y_test)
            all_y_pred_proba.append(y_pred_proba)

        self.all_y_test = np.concatenate(all_y_test)
        self.all_y_pred_proba = np.concatenate(all_y_pred_proba)

        fpr, tpr, _ = roc_curve(self.all_y_test, self.all_y_pred_proba)
        roc_auc = roc_auc_score(self.all_y_test, self.all_y_pred_proba)
        
        # Set file paths for saving plots
        if save_plots and output_dir:
            si_plot_path = f'{output_dir}/svm_si_plot.png'
            simple_plot_path = f'{output_dir}/svm_simple_plot.png'
        else:
            si_plot_path = None
            simple_plot_path = None

        self.plot_si_grad(save_plot=save_plots, file_path=si_plot_path)
        self.plot_simple(save_plot=save_plots, file_path=simple_plot_path)

        return roc_auc, self.all_y_test, self.all_y_pred_proba
