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
        self.fitted = {}  # stores params & masks

    def fit(self, spectra: np.ndarray, methods: list, y: np.ndarray = None):
        """Fit params for unsupervised steps and, if requested, the IntervalPLS mask.
           NOTE: iPLS selection is done on the unsupervised-preprocessed spectra.
        """
        self.fitted = {}

        # --- fit unsupervised pieces ---
        if 'EMSC' in methods:
            self.fitted['emsc_ref'] = np.mean(spectra, axis=1)

        if 'SNV' in methods:
            self.fitted['snv_mean'] = np.mean(spectra, axis=0, keepdims=True)
            std = np.std(spectra, axis=0, keepdims=True)
            # avoid divide-by-zero
            std = np.where(std == 0, 1.0, std)
            self.fitted['snv_std'] = std

        if 'Normalization' in methods:
            norm = np.linalg.norm(spectra, axis=0, keepdims=True)
            norm = np.where(norm == 0, 1.0, norm)
            self.fitted['norm'] = norm

        # --- build the unsupervised-preprocessed spectra for iPLS selection ---
        unsup_methods = [m for m in methods if m in ('EMSC', 'SNV', 'Normalization', 'Second Derivative')]
        X_unsup = self.transform(spectra.copy(), unsup_methods)  # (n_features, n_spectra)

        # --- fit IntervalPLS on the same space the model will use ---
        if 'IntervalPLS' in methods:
            if y is None:
                raise ValueError("`y` must be provided for IntervalPLS selection")
            # select_intervals_by_pls_r2 expects (n_samples, n_features)
            ipls_mask = self.select_intervals_by_pls_r2(
                X_unsup.T, y, n_intervals=150, n_components=2, cv_folds=5,
                threshold=self.ipls_threshold
            )
            self.fitted['ipls_mask'] = ipls_mask

    def transform(self, spectra: np.ndarray, methods: list):
        """Apply transforms in the usual order; apply iPLS mask last."""
        if 'EMSC' in methods:
            ref = self.fitted.get('emsc_ref')
            if ref is None:
                raise RuntimeError("EMSC parameters not fitted. Call fit(...) first.")
            emsc_out = np.zeros_like(spectra)
            for i in range(spectra.shape[1]):
                p = np.polyfit(ref, spectra[:, i], 1)
                emsc_out[:, i] = (spectra[:, i] - p[1]) / p[0]
            spectra = emsc_out

        if 'SNV' in methods:
            mean = self.fitted.get('snv_mean')
            std  = self.fitted.get('snv_std')
            if mean is None or std is None:
                raise RuntimeError("SNV parameters not fitted. Call fit(...) first.")
            spectra = (spectra - mean) / std

        if 'Normalization' in methods:
            norm = self.fitted.get('norm')
            if norm is None:
                raise RuntimeError("Normalization parameters not fitted. Call fit(...) first.")
            spectra = spectra / norm

        if 'Second Derivative' in methods:
            spectra = savgol_filter(spectra, window_length=11, polyorder=2, deriv=2, axis=0)

        if 'IntervalPLS' in methods:
            sel = self.fitted.get('ipls_mask')
            if sel is None:
                raise RuntimeError("IntervalPLS mask not fitted. Call fit(..., methods including 'IntervalPLS', y=...) first.")
            sel = np.asarray(sel, dtype=int)
            spectra = spectra[sel, :]

        return spectra

    def fit_transform(self, spectra: np.ndarray, methods: list, y: np.ndarray = None):
        self.fit(spectra, methods, y)
        return self.transform(spectra, methods)

    def select_intervals_by_pls_r2(self, X, y, n_intervals=150, n_components=2, cv_folds=5, threshold=0.0):
        """Return a 1D int array of selected feature indices."""
        n_samples, n_features = X.shape
        intervals = np.array_split(np.arange(n_features, dtype=int), n_intervals)
        selected = []
        r2_scores = []
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for inds in intervals:
            Xi = X[:, inds]  # (n_samples, len(inds))
            y_pred = np.zeros_like(y, dtype=float)
            for tr, va in kf.split(Xi):
                pls = PLSRegression(n_components=n_components)
                pls.fit(Xi[tr], y[tr])
                y_pred[va] = pls.predict(Xi[va]).ravel()
            r2 = r2_score(y, y_pred)
            r2_scores.append(r2)
            if r2 > threshold:
                selected.append(inds)

        # Report & visualize (closed afterwards to avoid Agg warnings)
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
        plt.close(fig)  # important for non-interactive backends

        if not selected:
            print("IntervalPLS: no intervals passed threshold, using full spectrum")
            return np.arange(n_features, dtype=int)

        return np.hstack(selected).astype(int)
