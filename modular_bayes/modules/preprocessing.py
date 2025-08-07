import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl

class Preprocessing:
    def __init__(self, ipls_threshold: float = 0.0):
        self.ipls_threshold = ipls_threshold
        self.fitted = {}  # Dictionary to store fitted parameters

    def fit(self, spectra: np.ndarray, methods: list, y: np.ndarray = None):
        self.fitted = {}  # Reset before each new fit

        if 'EMSC' in methods:
            self.fitted['emsc_ref'] = np.mean(spectra, axis=1)

        if 'SNV' in methods:
            self.fitted['snv_mean'] = np.mean(spectra, axis=0, keepdims=True)
            self.fitted['snv_std'] = np.std(spectra, axis=0, keepdims=True)

        if 'IntervalPLS' in methods:
            if y is None:
                raise ValueError("`y` must be provided for IntervalPLS selection")
            X = spectra.T
            self.fitted['ipls_mask'] = self.select_intervals_by_pls_r2(
                X, y, n_intervals=150, n_components=2, cv_folds=5,
                threshold=self.ipls_threshold
            )

        if 'Normalization' in methods:
            norm = np.linalg.norm(spectra, axis=0, keepdims=True)
            norm[norm == 0] = 1
            self.fitted['norm'] = norm

    def transform(self, spectra: np.ndarray, methods: list):
        if 'EMSC' in methods:
            ref = self.fitted.get('emsc_ref')
            emsc_out = np.zeros_like(spectra)
            for i in range(spectra.shape[1]):
                p = np.polyfit(ref, spectra[:, i], 1)
                emsc_out[:, i] = (spectra[:, i] - p[1]) / p[0]
            spectra = emsc_out

        if 'SNV' in methods:
            mean = self.fitted.get('snv_mean')
            std = self.fitted.get('snv_std')
            spectra = (spectra - mean) / std

        if 'Normalization' in methods:
            spectra = spectra / self.fitted['norm']

        if 'Second Derivative' in methods:
            spectra = savgol_filter(spectra, window_length=11, polyorder=2, deriv=2, axis=0)

        if 'IntervalPLS' in methods:
            sel = self.fitted.get('ipls_mask')
            spectra = spectra[sel, :]

        return spectra

    def fit_transform(self, spectra: np.ndarray, methods: list, y: np.ndarray = None):
        self.fit(spectra, methods, y)
        return self.transform(spectra, methods)

    def select_intervals_by_pls_r2(self, X, y, n_intervals=150, n_components=2, cv_folds=5, threshold=0.0):
        n_samples, n_features = X.shape
        intervals = np.array_split(np.arange(n_features), n_intervals)
        selected = []
        r2_scores = []
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

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

        # Visualization
        num_selected = len(selected)
        print(f"IntervalPLS: selected {num_selected}/{n_intervals} intervals (threshold={threshold})")

        fig, (ax_spec, ax_map) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for i in range(n_samples):
            ax_spec.plot(np.arange(n_features), X[i, :], color='gray', alpha=0.3)
        for block in intervals:
            ax_spec.axvline(block[0], color='black', linestyle='--', linewidth=0.5)
        ax_spec.set_ylabel("Intensity")
        ax_spec.set_title("Training spectra with interval boundaries")

        mean_spec = X.mean(axis=0)
        cmap = mpl.cm.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=min(r2_scores), vmax=max(r2_scores))
        for inds, r2 in zip(intervals, r2_scores):
            ax_map.plot(inds, mean_spec[inds], color=cmap(norm(r2)), linewidth=2)
        ax_map.set_xlabel("Feature index")
        ax_map.set_ylabel("Mean intensity")
        ax_map.set_title("Interval-wise R² mapping")

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(r2_scores)
        fig.colorbar(sm, ax=ax_map, label='Interval R²', orientation='vertical')
        plt.tight_layout()
        plt.show()

        if not selected:
            print("IntervalPLS: no intervals passed threshold, using full spectrum")
            return np.arange(n_features)

        return np.hstack(selected)
