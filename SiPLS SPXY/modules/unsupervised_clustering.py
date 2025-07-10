import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import os
from modules.preprocessing import Preprocessing
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

class UnsupervisedClustering:
    def __init__(self, y_label_df, spectra, output_dir):
        self.y_label_df = pd.DataFrame(np.repeat(y_label_df.values, 9, axis=0), columns=y_label_df.columns)
        self.spectra = spectra
        self.output_dir = output_dir

    def perform_pca_and_plot(self, x_array, plot_name, save_plot=False, preprocessing=None, pc_axes=(1, 2),
                             x_bounds=None, y_bounds=None, font_size=42, title_font_size=50,
                             axes_font_size=24, legend_font_size=30, offset_text_font_size=18,
                             custom_title_prefix=None):

        """Perform PCA on the given data and plot the results."""
        
        # Apply preprocessing if specified
        if preprocessing is not None:
            preprocess = Preprocessing(x_array)  # Pass the spectra (x_array) when initializing Preprocessing
            x_array = preprocess.preprocess(methods=preprocessing)
            
        # Perform PCA
        pca = PCA(n_components=max(pc_axes))
        principal_components = pca.fit_transform(np.transpose(x_array))
        explained_variance = pca.explained_variance_ratio_ * 100

        # Select which PCs to plot
        pc_x = f'PC{pc_axes[0]}'
        pc_y = f'PC{pc_axes[1]}'
        
        self.y_label_df[pc_x] = principal_components[:, pc_axes[0] - 1]
        self.y_label_df[pc_y] = principal_components[:, pc_axes[1] - 1]
    
        colors = {
            'Control': 'blue',
            'DM1': 'orange',
            'DM2': 'green',
            'DMD': 'purple'
        }
    
        fig, ax = plt.subplots(figsize=(16, 10))
        markers = {'Train': 'o', 'Test': 's'}  # circle for Train, square for Test
        colors = {'Control': 'blue', 'DM1': 'orange', 'DM2': 'green', 'DMD': 'purple'}
        
        for sample_type in self.y_label_df['Type'].unique():
            for data_set in ['Train', 'Test']:
                subset = self.y_label_df[(self.y_label_df['Type'] == sample_type) & (self.y_label_df['Set'] == data_set)]
                if not subset.empty:
                    if data_set == 'Test':
                        color = 'red'  # 🌟 Change test set color here
                    else:
                        color = colors.get(sample_type, 'gray')
        
                    ax.scatter(
                        subset[pc_x], subset[pc_y],
                        label=f'{sample_type} ({data_set})',
                        marker=markers[data_set],
                        color=color,
                        s=50
                    )

        ax.set_xlabel(f'Principal Component {pc_axes[0]} ({explained_variance[pc_axes[0] - 1]:.2f}%)',
                      fontsize=font_size, fontweight='bold', labelpad=15)
        ax.set_ylabel(f'Principal Component {pc_axes[1]} ({explained_variance[pc_axes[1] - 1]:.2f}%)',
                      fontsize=font_size, fontweight='bold', labelpad=15)
    
        if x_bounds:
            ax.set_xlim(x_bounds)
        if y_bounds:
            ax.set_ylim(y_bounds)
    
        legend_properties = {'size': legend_font_size, 'weight': 'bold'}
        ax.legend(prop=legend_properties)
    
        ax.tick_params(axis='x', labelsize=axes_font_size, width=2, which='major', length=5)
        ax.tick_params(axis='y', labelsize=axes_font_size, width=2, which='major', length=5)
    
        # Apply scientific notation to both axes
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        
        # Increase the font size of the scientific notation
        ax.xaxis.offsetText.set_fontsize(offset_text_font_size)
        ax.yaxis.offsetText.set_fontsize(offset_text_font_size)
    
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.grid(False)
        plt.tight_layout(rect=[0, 0, 1, 1.05])  # Add extra space at the top
    
        # Set a blank title to prevent the y-axis label from being cut off
        prefix = f"{custom_title_prefix} - " if custom_title_prefix else ""
        if preprocessing:
            title = f"{prefix}PCA (PC{pc_axes[0]} vs PC{pc_axes[1]}) - {' + '.join(preprocessing)}"
        else:
            title = f"{prefix}PCA (PC{pc_axes[0]} vs PC{pc_axes[1]}) - No Preprocessing"
        ax.set_title(title, fontsize=title_font_size, fontweight='bold')

    
        if save_plot:
            os.makedirs(self.output_dir, exist_ok=True)
            # Use the plot_name in the output path
            output_path = os.path.join(self.output_dir, f'{plot_name}.png')
            plt.savefig(output_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.2)
    
        plt.show()
        
    def fit_pca_on_train(self, x_train, preprocessing=None, n_components=3):
        if preprocessing:
            x_train = Preprocessing(x_train).preprocess(methods=preprocessing)
    
        pca = PCA(n_components=n_components)
        train_scores = pca.fit_transform(x_train.T)
    
        return pca, train_scores


    def perform_pca_and_plot_with_severity(self, x_array, plot_name, save_plot=False, preprocessing=None, pc_axes=(1, 2),
                                           font_size=42, title_font_size=50, axes_font_size=24, legend_font_size=30,
                                           offset_text_font_size=18, custom_title_prefix=None):

        """Perform PCA on the given data and plot the results with severity of DM1 samples."""
        
        # Apply preprocessing if specified
        if preprocessing is not None:
            preprocess = Preprocessing(x_array)  # Pass the spectra (x_array) when initializing Preprocessing
            x_array = preprocess.preprocess(methods=preprocessing)
        
        pca = PCA(n_components=max(pc_axes))
        principal_components = pca.fit_transform(np.transpose(x_array))
        explained_variance = pca.explained_variance_ratio_ * 100

        pc_x = f'PC{pc_axes[0]}'
        pc_y = f'PC{pc_axes[1]}'
    
        self.y_label_df[pc_x] = principal_components[:, pc_axes[0] - 1]
        self.y_label_df[pc_y] = principal_components[:, pc_axes[1] - 1]
    
        cmap = sns.color_palette("viridis", as_cmap=True)
    
        fig, ax = plt.subplots(figsize=(16, 10))
        scatter = ax.scatter(
            self.y_label_df[pc_x], self.y_label_df[pc_y], 
            c=self.y_label_df['target_SI'], cmap=cmap, s=50, alpha=0.7, edgecolors='w', linewidth=0.5
        )

        ax.set_xlabel(f'Principal Component {pc_axes[0]} ({explained_variance[pc_axes[0] - 1]:.2f}%)',
                      fontsize=font_size, fontweight='bold', labelpad=15)
        ax.set_ylabel(f'Principal Component {pc_axes[1]} ({explained_variance[pc_axes[1] - 1]:.2f}%)',
                      fontsize=font_size, fontweight='bold', labelpad=15)
    
        ax.tick_params(axis='x', labelsize=axes_font_size, width=2, which='major', length=5)
        ax.tick_params(axis='y', labelsize=axes_font_size, width=2, which='major', length=5)
    
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(offset_text_font_size)
        ax.yaxis.offsetText.set_fontsize(offset_text_font_size)
    
        plt.grid(False)
        
        cbar = plt.colorbar(scatter, pad=0.01)
        cbar.set_label('DM1 Severity (Splicing Index)', fontsize=font_size-10, fontweight='bold')
        cbar.ax.tick_params(labelsize=axes_font_size, width=2, which='major', length=5)
        
        plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
        plt.tight_layout(rect=[0, 0, 1, 1.05]) 
    
        # Set a blank title to prevent the y-axis label from being cut off
        prefix = f"{custom_title_prefix} - " if custom_title_prefix else ""
        if preprocessing:
            title = f"{prefix}PCA (PC{pc_axes[0]} vs PC{pc_axes[1]}) - {' + '.join(preprocessing)}"
        else:
            title = f"{prefix}PCA (PC{pc_axes[0]} vs PC{pc_axes[1]}) - No Preprocessing"
        ax.set_title(title, fontsize=title_font_size, fontweight='bold')

    
        if save_plot:
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, f'{plot_name}.png')
            plt.savefig(output_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.2)
    
        plt.show()
    def pca_vs_baseline_3d_with_severity(self, spectra, save_plot=False, plot_name="pca_vs_baseline_3d_with_severity"):
        """
        3D visualization of PCA components with baseline drift on Z-axis and SI value as color.
    
        Parameters:
            spectra (np.ndarray): 2D array (wavelengths × spectra).
            save_plot (bool): Whether to save the figure.
            plot_name (str): Filename to save the plot.
        """
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.signal import savgol_filter
        from scipy.stats import pearsonr
        import matplotlib.pyplot as plt
        import seaborn as sns
    
        # Estimate mean baseline using Savitzky-Golay filter per spectrum
        baseline_means = np.array([
            np.mean(savgol_filter(spectra[:, i], window_length=51, polyorder=3))
            for i in range(spectra.shape[1])
        ])
    
        # Perform PCA
        pca = PCA(n_components=2)
        pca_scores = pca.fit_transform(spectra.T)
        pc1 = pca_scores[:, 0]
        pc2 = pca_scores[:, 1]
    
        # Get SI values (repeated per spectrum)
        if "target_SI" not in self.y_label_df.columns:
            raise ValueError("Expected 'target_SI' column in metadata.")
        si_values = self.y_label_df['target_SI'].values
    
        # Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        cmap = sns.color_palette("viridis", as_cmap=True)
        sc = ax.scatter(pc1, pc2, baseline_means, c=si_values, cmap=cmap, s=50, edgecolor='k', alpha=0.85)
        
        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
        ax.set_zlabel("Baseline Mean (SG Filter)", fontsize=12)
        ax.set_title("PCA vs Baseline with SI Gradient", fontsize=14)
    
        cbar = plt.colorbar(sc, pad=0.02)
        cbar.set_label("Splicing Index (SI)", fontsize=12)
    
        plt.tight_layout()
    
        # Save if needed
        if save_plot:
            os.makedirs(self.output_dir, exist_ok=True)
            save_path = os.path.join(self.output_dir, f"{plot_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
        plt.show()
    
        # Print correlation analysis
        r1, p1 = pearsonr(baseline_means, pc1)
        r2, p2 = pearsonr(baseline_means, pc2)
        print(f"📈 Baseline vs PC1 → r = {r1:.3f}, p = {p1:.4f}")
        print(f"📈 Baseline vs PC2 → r = {r2:.3f}, p = {p2:.4f}")
        
    def transform_test_with_pca(self, pca, x_test, y_meta_test, preprocessing=None, pc_axes=(1, 2),
                                plot_name="test_projection", save_plot=False, custom_title_prefix=None,
                                target_column=None):
        if preprocessing:
            x_test = Preprocessing(x_test).preprocess(methods=preprocessing)
    
        test_scores = pca.transform(x_test.T)
        explained_variance = pca.explained_variance_ratio_ * 100
    
        pc_x = f'PC{pc_axes[0]}'
        pc_y = f'PC{pc_axes[1]}'
    
        # 🔁 Repeat metadata to match 9 spectra per sample
        y_meta_test = pd.DataFrame(np.repeat(y_meta_test.values, 9, axis=0), columns=y_meta_test.columns)
    
        # ✅ Assign PCA scores
        y_meta_test[pc_x] = test_scores[:, pc_axes[0] - 1]
        y_meta_test[pc_y] = test_scores[:, pc_axes[1] - 1]
    
        # ─── Plot ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(16, 10))
    
        if target_column:
            if target_column not in y_meta_test.columns:
                raise ValueError(f"Target column '{target_column}' not found in test metadata.")
    
            sc = ax.scatter(
                y_meta_test[pc_x],
                y_meta_test[pc_y],
                c=y_meta_test[target_column],
                cmap='viridis',
                edgecolor='k',
                s=60
            )
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(target_column, fontsize=18)
        else:
            for sample_type in y_meta_test['Type'].unique():
                subset = y_meta_test[y_meta_test['Type'] == sample_type]
                ax.scatter(
                    subset[pc_x],
                    subset[pc_y],
                    label=sample_type,
                    color='red',  # Default to red for test set
                    marker='s',
                    s=50
                )
            ax.legend(fontsize=16)
    
        ax.set_xlabel(f'PC{pc_axes[0]} ({explained_variance[pc_axes[0] - 1]:.2f}%)', fontsize=24, fontweight='bold')
        ax.set_ylabel(f'PC{pc_axes[1]} ({explained_variance[pc_axes[1] - 1]:.2f}%)', fontsize=24, fontweight='bold')
        ax.set_title(f"{custom_title_prefix or ''} Test Set PCA Projection", fontsize=28, fontweight='bold')
    
        if save_plot:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, f"{plot_name}.png"), dpi=600, bbox_inches='tight')
    
        plt.show()
    

# THIS IS FOR ADF AND HGS
    def perform_pca_and_plot_with_target(self, x_array, plot_name, target_column, save_plot=False,
                                         preprocessing=None, pc_axes=(1, 2), font_size=42,
                                         title_font_size=50, axes_font_size=24, legend_font_size=30,
                                         offset_text_font_size=18, custom_title_prefix=None):
        """Perform PCA and color by the specified target column (e.g., ADF, HGS)."""
    
        if preprocessing is not None:
            preprocess = Preprocessing(x_array)
            x_array = preprocess.preprocess(methods=preprocessing)
    
        pca = PCA(n_components=max(pc_axes))
        principal_components = pca.fit_transform(x_array.T)
        explained_variance = pca.explained_variance_ratio_ * 100
    
        pc_x = f'PC{pc_axes[0]}'
        pc_y = f'PC{pc_axes[1]}'
    
        self.y_label_df[pc_x] = principal_components[:, pc_axes[0] - 1]
        self.y_label_df[pc_y] = principal_components[:, pc_axes[1] - 1]
    
        if target_column not in self.y_label_df.columns:
            raise ValueError(f"Expected '{target_column}' column in metadata.")
    
        target_values = self.y_label_df[target_column].values
    
        cmap = sns.color_palette("viridis", as_cmap=True)
    
        fig, ax = plt.subplots(figsize=(16, 10))
        scatter = ax.scatter(
            self.y_label_df[pc_x], self.y_label_df[pc_y],
            c=target_values, cmap=cmap, s=50, alpha=0.7, edgecolors='w', linewidth=0.5
        )
    
        ax.set_xlabel(f'Principal Component {pc_axes[0]} ({explained_variance[pc_axes[0] - 1]:.2f}%)',
                      fontsize=font_size, fontweight='bold', labelpad=15)
        ax.set_ylabel(f'Principal Component {pc_axes[1]} ({explained_variance[pc_axes[1] - 1]:.2f}%)',
                      fontsize=font_size, fontweight='bold', labelpad=15)
    
        ax.tick_params(axis='x', labelsize=axes_font_size, width=2)
        ax.tick_params(axis='y', labelsize=axes_font_size, width=2)
    
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        ax.xaxis.offsetText.set_fontsize(offset_text_font_size)
        ax.yaxis.offsetText.set_fontsize(offset_text_font_size)
    
        cbar = plt.colorbar(scatter, pad=0.01)
        cbar.set_label(target_column + " (Scaled)", fontsize=font_size - 10, fontweight='bold')
        cbar.ax.tick_params(labelsize=axes_font_size, width=2)
    
        plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
        plt.tight_layout(rect=[0, 0, 1, 1.05])
    
        prefix = f"{custom_title_prefix} - " if custom_title_prefix else ""
        if preprocessing:
            title = f"{prefix}PCA ({target_column}) (PC{pc_axes[0]} vs PC{pc_axes[1]}) - {' + '.join(preprocessing)}"
        else:
            title = f"{prefix}PCA ({target_column}) (PC{pc_axes[0]} vs PC{pc_axes[1]}) - No Preprocessing"
    
        ax.set_title(title, fontsize=title_font_size, fontweight='bold')
    
        if save_plot:
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, f'{plot_name}.png')
            plt.savefig(output_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.2)
    
        plt.show()
        
    def transform_test_with_pca_with_target(self, pca, x_test, y_meta_test, target_column,
                                            preprocessing=None, pc_axes=(1, 2),
                                            plot_name="test_projection_with_target",
                                            save_plot=False, custom_title_prefix=None,
                                            font_size=42, title_font_size=50,
                                            axes_font_size=24, legend_font_size=30,
                                            offset_text_font_size=18):
        """Apply a trained PCA model to test data and color the plot by a target variable."""

        if preprocessing:
            x_test = Preprocessing(x_test).preprocess(methods=preprocessing)

        test_scores = pca.transform(x_test.T)
        explained_variance = pca.explained_variance_ratio_ * 100

        pc_x = f'PC{pc_axes[0]}'
        pc_y = f'PC{pc_axes[1]}'

        y_meta_test = pd.DataFrame(np.repeat(y_meta_test.values, 9, axis=0), columns=y_meta_test.columns)
        y_meta_test[pc_x] = test_scores[:, pc_axes[0] - 1]
        y_meta_test[pc_y] = test_scores[:, pc_axes[1] - 1]

        if target_column not in y_meta_test.columns:
            raise ValueError(f"Expected '{target_column}' column in metadata.")

        target_values = y_meta_test[target_column].values
        cmap = sns.color_palette("viridis", as_cmap=True)

        fig, ax = plt.subplots(figsize=(16, 10))
        scatter = ax.scatter(
            y_meta_test[pc_x], y_meta_test[pc_y],
            c=target_values, cmap=cmap, s=50, alpha=0.7, edgecolors='w', linewidth=0.5
        )

        ax.set_xlabel(f'PC{pc_axes[0]} ({explained_variance[pc_axes[0] - 1]:.2f}%)', fontsize=font_size, fontweight='bold', labelpad=15)
        ax.set_ylabel(f'PC{pc_axes[1]} ({explained_variance[pc_axes[1] - 1]:.2f}%)', fontsize=font_size, fontweight='bold', labelpad=15)
        ax.tick_params(axis='x', labelsize=axes_font_size, width=2)
        ax.tick_params(axis='y', labelsize=axes_font_size, width=2)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        ax.xaxis.offsetText.set_fontsize(offset_text_font_size)
        ax.yaxis.offsetText.set_fontsize(offset_text_font_size)

        cbar = plt.colorbar(scatter, pad=0.01)
        cbar.set_label(target_column + " (Scaled)", fontsize=font_size - 10, fontweight='bold')
        cbar.ax.tick_params(labelsize=axes_font_size, width=2)

        plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
        plt.tight_layout(rect=[0, 0, 1, 1.05])

        prefix = f"{custom_title_prefix} - " if custom_title_prefix else ""
        if preprocessing:
            title = f"{prefix}Test Set PCA ({target_column}) (PC{pc_axes[0]} vs PC{pc_axes[1]}) - {' + '.join(preprocessing)}"
        else:
            title = f"{prefix}Test Set PCA ({target_column}) (PC{pc_axes[0]} vs PC{pc_axes[1]}) - No Preprocessing"

        ax.set_title(title, fontsize=title_font_size, fontweight='bold')

        if save_plot:
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, f'{plot_name}.png')
            plt.savefig(output_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.2)

        plt.show()
