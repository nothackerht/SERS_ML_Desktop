import numpy as np
from scipy.signal import savgol_filter

class Preprocessing:
    def __init__(self, spectra):
        """
        Initialize the Preprocessing class with the spectra.

        Parameters:
        spectra (np.ndarray): The spectra to preprocess, where rows represent wavelengths and columns represent different spectra.
        """
        self.spectra = spectra

    def emsc(self, spectra, reference=None):
        """Extended Multiplicative Signal Correction (EMSC)."""
        if reference is None:
            reference = np.mean(spectra, axis=1)  # Calculate reference across spectra (columns)
        spectra_emsc = np.zeros_like(spectra)
        for i in range(spectra.shape[1]):  # Iterate over each spectrum
            p = np.polyfit(reference, spectra[:, i], 1)  # Apply polyfit along the wavelengths (rows)
            spectra_emsc[:, i] = (spectra[:, i] - p[1]) / p[0]
        return spectra_emsc

    def normalize_spectrum(self, spectra):
        """Normalize the spectrum to unit length."""
        norm = np.linalg.norm(spectra, axis=0, keepdims=True)  # Normalize each spectrum (column) along the wavelengths (rows)
        norm[norm == 0] = 1  # To avoid division by zero
        return spectra / norm

    def snv(self, spectra):
        """Standard Normal Variate (SNV) transformation."""
        spectra_snv = (spectra - np.mean(spectra, axis=0, keepdims=True)) / np.std(spectra, axis=0, keepdims=True)  # Apply SNV across wavelengths for each spectrum
        return spectra_snv

    def second_derivative(self, spectra):
        """Calculate the second derivative of the spectra."""
        return savgol_filter(spectra, window_length=11, polyorder=2, deriv=2, axis=0)  # Apply derivative along wavelengths (rows)

    def preprocess(self, methods):
        """
        Apply the specified preprocessing methods to the spectra.

        Parameters:
        methods (list): A list of preprocessing methods to apply, in the order they should be applied.

        Returns:
        np.ndarray: The preprocessed spectra.
        """
        spectra = self.spectra.copy()
        for method in methods:
            if method == 'EMSC':
                spectra = self.emsc(spectra)
            elif method == 'Normalization':
                spectra = self.normalize_spectrum(spectra)
            elif method == 'SNV':
                spectra = self.snv(spectra)
            elif method == 'Second Derivative':
                spectra = self.second_derivative(spectra)
            else:
                raise ValueError(f"Unknown preprocessing method: {method}")
        return spectra
