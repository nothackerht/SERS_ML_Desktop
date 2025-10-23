import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import os
from modules.preprocessing import Preprocessing

class UnsupervisedClustering:
    def __init__(self, y_label_df, spectra, output_dir):
        self.y_label_df = pd.DataFrame(np.repeat(y_label_df.values, 9, axis=0), columns=y_label_df.columns)
        self.spectra = spectra
        self.output_dir = output_dir

    def perform_pca_and_plot(self, x_array, plot_name, save_plot=False, preprocessing=None, pc_axes=(1, 2),
                             x_bounds=None, y_bounds=None, font_size=42, title_font_size=50,
                             axes_font_size=24, legend_font_size=30, offset_text_font_size=18):
        """Perform PCA on the given data and plot the results."""
        
        # Apply preprocessing if specified
        if preprocessing:
            pre = Preprocessing()
            pre.fit(x_array, methods=preprocessing)
            x_array = pre.transform(x_array, methods=preprocessing)
            
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
        for label in self.y_label_df['Type'].unique():
            subset = self.y_label_df[self.y_label_df['Type'] == label]
            ax.scatter(subset[pc_x], subset[pc_y], label=label, color=colors[label], s=50)
    
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
        ax.set_title(' ', fontsize=title_font_size, fontweight='bold')
    
        if save_plot:
            os.makedirs(self.output_dir, exist_ok=True)
            # Use the plot_name in the output path
            output_path = os.path.join(self.output_dir, f'{plot_name}.png')
            plt.savefig(output_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.2)
    
        plt.show()

    def perform_pca_and_plot_with_severity(self, x_array, plot_name, save_plot=False, preprocessing=None, pc_axes=(1, 2),
                                           font_size=42, title_font_size=50, axes_font_size=24, legend_font_size=30,
                                           offset_text_font_size=18):
        """Perform PCA on the given data and plot the results with severity of DM1 samples."""
        
        # Apply preprocessing if specified
        if preprocessing:
            pre = Preprocessing()                     # e.g., Preprocessing(ipls_threshold=0.0)
            pre.fit(x_array, methods=preprocessing)   # fit params (e.g., EMSC ref) ON THIS DATA
            x_array = pre.transform(x_array, methods=preprocessing)
                
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
        ax.set_title(' ', fontsize=title_font_size, fontweight='bold')
    
        if save_plot:
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, f'{plot_name}.png')
            plt.savefig(output_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.2)
    
        plt.show()
