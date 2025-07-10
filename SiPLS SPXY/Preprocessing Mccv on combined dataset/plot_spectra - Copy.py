import matplotlib.pyplot as plt

def create_plot(wavenumber, all_spectra, y_label_df, output_path):
    # Define categories
    categories = ['DM1', 'Control', 'DMD', 'DM2']

    # Filter spectra based on categories
    spectra_dict = {}
    for category in categories:
        indices = y_label_df[y_label_df['Type'] == category].index
        spectra_dict[category] = all_spectra[:, indices]

    # Define colors for the categories
    colors = {
        'DM1': 'blue',
        'Control': 'green',
        'DMD': 'red',
        'DM2': 'purple'
    }

    # Define the break point for the two regions
    break_point = 967

    # Function to plot individual spectra
    def plot_spectra(ax, wavenumber, spectra, color, category):
        # Plot individual spectra for the first region
        for i in range(spectra.shape[1]):
            spectrum = spectra[:break_point, i]
            ax.plot(wavenumber[:break_point], spectrum, color=color, linewidth=1.5)
            
        # Plot individual spectra for the second region
        for i in range(spectra.shape[1]):
            spectrum = spectra[break_point+1:, i]
            ax.plot(wavenumber[break_point+1:], spectrum, color=color, linewidth=1.5)
    
        ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=30, fontweight='bold')
        ax.set_ylabel('Intensity (A.U)', fontsize=30, fontweight='bold')
        ax.legend([category], loc='upper right', prop={'size': 30, 'weight': 'bold'})


    # Main function to create the plot
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 20  # Set font size for x-axis numerical labels
    plt.rcParams['ytick.labelsize'] = 20  # Set font size for y-axis numerical labels
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    # Plot data for each category
    for ax, category in zip(axs.flatten(), categories):
        spectra = spectra_dict[category]
        plot_spectra(ax, wavenumber, spectra, colors[category], category)

    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.show()
def plot_train_test_spectra(wavenumbers, train_spectra, test_spectra, output_path=None):
    """
    Overlay training and test spectra with a discontinuity between 1800–2400 cm⁻¹.
    Left = 0–1800 cm⁻¹, Right = 2400–max cm⁻¹.
    """
    # Define the mask regions
    left_mask = wavenumbers <= 1800
    right_mask = wavenumbers >= 2400

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True, gridspec_kw={'wspace': 0.05})

    # Plot Left Region (0–1800 cm⁻¹)
    for i in range(train_spectra.shape[1]):
        ax1.plot(wavenumbers[left_mask], train_spectra[left_mask, i], color='blue', alpha=0.2)
    for i in range(test_spectra.shape[1]):
        ax1.plot(wavenumbers[left_mask], test_spectra[left_mask, i], color='red', alpha=0.6)

    # Plot Right Region (2400–end cm⁻¹)
    for i in range(train_spectra.shape[1]):
        ax2.plot(wavenumbers[right_mask], train_spectra[right_mask, i], color='blue', alpha=0.2)
    for i in range(test_spectra.shape[1]):
        ax2.plot(wavenumbers[right_mask], test_spectra[right_mask, i], color='red', alpha=0.6)

    # Shared labels
    ax1.set_ylabel("Intensity (a.u.)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=14, fontweight='bold')

    # Set axis formatting
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)
    ax1.set_xlim([wavenumbers[left_mask][0], wavenumbers[left_mask][-1]])
    ax2.set_xlim([wavenumbers[right_mask][0], wavenumbers[right_mask][-1]])

    # Add title
    fig.suptitle("Overlay of Training (Blue) and Test (Red) Spectra", fontsize=16, fontweight='bold')

    # Finalize layout
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
