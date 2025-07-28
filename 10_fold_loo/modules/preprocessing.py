import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl

class Preprocessing:
    def __init__(self, spectra: np.ndarray, ipls_threshold: float = 0.0):
        """
        Initialize the Preprocessing pipeline.

        Parameters:
        -----------
        spectra : np.ndarray of shape (n_wavenumbers, n_spectra)
            Your raw spectra matrix (each column is one spectrum).

        ipls_threshold : float, default=0.0
            Minimum cross-validated R² to keep an interval in IntervalPLS.
        """
        # ensure we’re working with a proper numpy array
        self.spectra = np.asarray(spectra)
        # store the user’s IntervalPLS R² cutoff
        self.ipls_threshold = ipls_threshold


    def emsc(self, spectra, reference=None):
        """Extended Multiplicative Signal Correction (EMSC)."""
        if reference is None:
            reference = np.mean(spectra, axis=1)
        spectra_emsc = np.zeros_like(spectra)
        for i in range(spectra.shape[1]):
            p = np.polyfit(reference, spectra[:, i], 1)
            spectra_emsc[:, i] = (spectra[:, i] - p[1]) / p[0]
        return spectra_emsc

    def normalize_spectrum(self, spectra):
        """Normalize the spectrum to unit length."""
        norm = np.linalg.norm(spectra, axis=0, keepdims=True)
        norm[norm == 0] = 1
        return spectra / norm

    def snv(self, spectra):
        """Standard Normal Variate (SNV) transformation."""
        return (spectra - np.mean(spectra, axis=0, keepdims=True)) \
               / np.std(spectra, axis=0, keepdims=True)

    def second_derivative(self, spectra):
        """Calculate the second derivative of the spectra."""
        return savgol_filter(spectra, window_length=11, polyorder=2, deriv=2, axis=0)

    def select_intervals_by_pls_r2(
        self,
        X,
        y,
        n_intervals=150,
        n_components=2,
        cv_folds=5,
        threshold=0.0
    ):
        """
        Split the spectrum into intervals, run PLS on each, and select intervals
        whose cross-validated R² exceeds the threshold, then plot:

        1) Top: overlay of all raw spectra with vertical lines at interval boundaries
        2) Bottom: mean spectrum colored by interval R², with a colorbar legend

        Parameters:
        X (np.ndarray): shape (n_samples, n_features) spectra matrix.
        y (np.ndarray): shape (n_samples,) target values.
        n_intervals (int): Number of contiguous intervals to split the features into.
        n_components (int): Number of PLS components for evaluation.
        cv_folds (int): Number of folds for inner CV.
        threshold (float): Minimum R² for interval selection.

        Returns:
        np.ndarray: indices of features to keep.
        """
        n_samples, n_features = X.shape
        # 1) split feature indices into contiguous blocks
        intervals = np.array_split(np.arange(n_features), n_intervals)
        selected = []
        r2_scores = []
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # 2) evaluate each interval via inner CV
        for inds in intervals:
            Xi = X[:, inds]
            y_pred = np.zeros_like(y, dtype=float)
            for tr, va in kf.split(Xi):
                pls = PLSRegression(n_components=n_components)
                pls.fit(Xi[tr], y[tr])
                y_pred[va] = pls.predict(Xi[va]).ravel()
            r2 = r2_score(y, y_pred)
            r2_scores.append(r2)
            if r2 > threshold:
                selected.append(inds)

        # 3) summary print
        num_selected = len(selected)
        print(f"IntervalPLS: selected {num_selected}/{n_intervals} intervals (threshold={threshold})")

        # 4) build the two‐panel figure
        fig, (ax_spec, ax_map) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 4a) Top: overlay all raw training spectra
        for i in range(n_samples):
            ax_spec.plot(np.arange(n_features), X[i, :], color='gray', alpha=0.3)
        # add interval boundaries
        for block in intervals:
            ax_spec.axvline(block[0], color='black', linestyle='--', linewidth=0.5)
        ax_spec.set_ylabel("Intensity")
        ax_spec.set_title("Training spectra with interval boundaries")

        # 4b) Bottom: mean spectrum colored by R²
        mean_spec = X.mean(axis=0)
        cmap = mpl.cm.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=min(r2_scores), vmax=max(r2_scores))
        for inds, r2 in zip(intervals, r2_scores):
            ax_map.plot(inds, mean_spec[inds], color=cmap(norm(r2)), linewidth=2)
        ax_map.set_xlabel("Feature index")
        ax_map.set_ylabel("Mean intensity")
        ax_map.set_title("Interval-wise R² mapping")

        # 4c) colorbar legend
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(r2_scores)
        fig.colorbar(sm, ax=ax_map, label='Interval R²', orientation='vertical')

        plt.tight_layout()
        plt.show()

        # 5) fallback if none passed
        if not selected:
            print("IntervalPLS: no intervals passed threshold, using full spectrum")
            return np.arange(n_features)

        # 6) return flattened list of all selected feature indices
        return np.hstack(selected)


    def preprocess(self, methods, y=None):
        """
        Apply the specified preprocessing methods to the spectra.
        
        methods: list of strings, e.g. ['SNV','IntervalPLS']
        y      : sample-level targets (needed by IntervalPLS)
        """
        # ─── NEW: coerce y into an np.ndarray if it isn’t one ──────────
        if y is not None and not isinstance(y, np.ndarray):
            y = np.asarray(y)
        # ────────────────────────────────────────────────────────────────

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
            elif method == 'IntervalPLS':
                if y is None:
                    raise ValueError("`y` must be provided for IntervalPLS selection")
                # transpose to (n_samples, n_features)
                X = spectra.T
                sel = self.select_intervals_by_pls_r2(
                    X,
                    y.ravel() if y.ndim > 1 else y,
                    n_intervals=150,
                    n_components=2,
                    cv_folds=5,
                    threshold=self.ipls_threshold
                )
                spectra = spectra[sel, :]
            else:
                raise ValueError(f"Unknown preprocessing method: {method}")
        return spectra

