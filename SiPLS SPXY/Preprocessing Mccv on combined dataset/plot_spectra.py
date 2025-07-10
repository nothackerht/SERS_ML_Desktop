import matplotlib.pyplot as plt
import numpy as np

def plot_train_test_spectra(wavenumbers, train_spectra, test_spectra, train_filenames, test_filenames, output_path=None):


    left_mask = wavenumbers <= 1800
    right_mask = wavenumbers >= 2400

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={'wspace': 0.05})

    for i in range(train_spectra.shape[1]):
        ax1.plot(wavenumbers[left_mask], train_spectra[left_mask, i], color='blue', alpha=0.2)
        ax2.plot(wavenumbers[right_mask], train_spectra[right_mask, i], color='blue', alpha=0.2)
        print(f"[TRAIN - FIRST] File: {train_filenames[i]} → Wavelengths: {np.sum(left_mask)}, Spectrum length: {np.sum(left_mask)}")
        print(f"[TRAIN - SECOND] File: {train_filenames[i]} → Wavelengths: {np.sum(right_mask)}, Spectrum length: {np.sum(right_mask)}")

    for i in range(test_spectra.shape[1]):
        ax1.plot(wavenumbers[left_mask], test_spectra[left_mask, i], color='red', alpha=0.6)
        ax2.plot(wavenumbers[right_mask], test_spectra[right_mask, i], color='red', alpha=0.6)
        print(f"[TEST - FIRST] File: {test_filenames[i]} → Wavelengths: {np.sum(left_mask)}, Spectrum length: {np.sum(left_mask)}")
        print(f"[TEST - SECOND] File: {test_filenames[i]} → Wavelengths: {np.sum(right_mask)}, Spectrum length: {np.sum(right_mask)}")

    ax1.set_ylabel("Intensity (a.u.)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=14, fontweight='bold')

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    ax1.set_xlim([wavenumbers[left_mask][0], wavenumbers[left_mask][-1]])
    ax2.set_xlim([wavenumbers[right_mask][0], wavenumbers[right_mask][-1]])

    fig.suptitle("Overlay of Training (Blue) and Test (Red) Spectra", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_plot(wavenumber, all_spectra, y_label_df, output_path, filenames_per_column):
    # Define categories
    categories = ['DM1', 'Control', 'DMD', 'DM2']

    # Filter spectra based on categories
    spectra_dict = {}
    filenames_dict = {}

    idx_counter = 0
    for category in categories:
        indices = y_label_df[y_label_df['Type'] == category].index
        spectra_dict[category] = all_spectra[:, indices]
        filenames_dict[category] = [filenames_per_column[i] for i in indices]

    # Define colors for the categories
    colors = {
        'DM1': 'blue',
        'Control': 'green',
        'DMD': 'red',
        'DM2': 'purple'
    }

    # Function to plot individual spectra
    def plot_spectra(ax, wavenumber, spectra, color, category, filenames):
        break_left = wavenumber <= 1800
        break_right = wavenumber >= 2400

        for i in range(spectra.shape[1]):
            spectrum_left = spectra[break_left, i]
            print(f"[{category} - FIRST] File: {filenames[i]} → Wavelengths: {len(wavenumber[break_left])}, Spectrum length: {spectrum_left.shape[0]}")
            ax.plot(wavenumber[break_left], spectrum_left, color=color, linewidth=1.5)

        for i in range(spectra.shape[1]):
            spectrum_right = spectra[break_right, i]
            print(f"[{category} - SECOND] File: {filenames[i]} → Wavelengths: {len(wavenumber[break_right])}, Spectrum length: {spectrum_right.shape[0]}")
            ax.plot(wavenumber[break_right], spectrum_right, color=color, linewidth=1.5)

        ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=30, fontweight='bold')
        ax.set_ylabel('Intensity (A.U)', fontsize=30, fontweight='bold')
        ax.legend([category], loc='upper right', prop={'size': 30, 'weight': 'bold'})

    # Main plotting section
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    for ax, category in zip(axs.flatten(), categories):
        spectra = spectra_dict[category]
        filenames = filenames_dict[category]
        plot_spectra(ax, wavenumber, spectra, colors[category], category, filenames)

    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.show()
