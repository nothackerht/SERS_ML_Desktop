import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl


def _r2_on_sample_means(y, yhat, groups):
    """
    Compute R² on per-sample means. `groups` repeats the sample id for each spectrum
    (e.g., 9 identical ids per sample if you have 9 replicates).
    """
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)
    groups = np.asarray(groups).reshape(-1)

    # aggregate within groups
    _, inv = np.unique(groups, return_inverse=True)
    n = inv.max() + 1
    y_g = np.zeros(n); yhat_g = np.zeros(n); counts = np.zeros(n)
    np.add.at(y_g, inv, y)
    np.add.at(yhat_g, inv, yhat)
    np.add.at(counts, inv, 1)
    y_g /= counts
    yhat_g /= counts
    return r2_score(y_g, yhat_g)


class Preprocessing:
    def __init__(self, ipls_threshold: float = 0.0):
        self.ipls_threshold = ipls_threshold
        self.fitted = {}  # stores params & masks

    # -------------------------
    # Unsupervised fit/transform
    # -------------------------
    def fit(self, spectra: np.ndarray, methods: list, y: np.ndarray = None, groups: np.ndarray = None):
        """
        Fit parameters for unsupervised steps. Optionally compute an IntervalPLS mask
        if 'IntervalPLS' is in methods and y (+ optionally groups) are provided.
        NOTE: iPLS selection is done on the *unsupervised-preprocessed* spectra.
        """
        self.fitted = {}

        # --- fit unsupervised pieces (spectra shape: [n_features, n_spectra]) ---
        if 'EMSC' in methods:
            self.fitted['emsc_ref'] = np.mean(spectra, axis=1)

        if 'SNV' in methods:
            self.fitted['snv_mean'] = np.mean(spectra, axis=0, keepdims=True)
            std = np.std(spectra, axis=0, keepdims=True)
            std = np.where(std == 0, 1.0, std)  # avoid divide-by-zero
            self.fitted['snv_std'] = std

        if 'Normalization' in methods:
            norm = np.linalg.norm(spectra, axis=0, keepdims=True)
            norm = np.where(norm == 0, 1.0, norm)
            self.fitted['norm'] = norm

        # --- build the unsupervised-preprocessed spectra for iPLS selection ---
        unsup_methods = [m for m in methods if m in ('EMSC', 'SNV', 'Normalization', 'Second Derivative')]
        X_unsup = self.transform(spectra.copy(), unsup_methods)  # (n_features, n_spectra)

        # --- optionally fit IntervalPLS on the same space the model will use ---
        if 'IntervalPLS' in methods and y is not None:
            if groups is not None:
                # grouped, leakage-free selector
                ipls_mask = self.select_intervals_grouped(
                    X_unsup.T, y, groups,
                    n_intervals=150, n_components=2, cv_folds=5,
                    select_mode="threshold", threshold=self.ipls_threshold
                )
            else:
                # legacy KFold variant (not grouped) – kept for compatibility
                ipls_mask = self.select_intervals_by_pls_r2(
                    X_unsup.T, y, n_intervals=150, n_components=2, cv_folds=5,
                    threshold=self.ipls_threshold
                )
            self.fitted['ipls_mask'] = ipls_mask

    def transform(self, spectra: np.ndarray, methods: list):
        """
        Apply transforms in the usual order; apply iPLS mask last.
        Expects any needed params to be in self.fitted (from fit()).
        """
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
                raise RuntimeError("IntervalPLS mask not fitted. "
                                   "Call fit(..., methods including 'IntervalPLS', y=..., groups=...) "
                                   "or set self.fitted['ipls_mask'] manually.")
            sel = np.asarray(sel, dtype=int)
            spectra = spectra[sel, :]

        return spectra

    def fit_transform(self, spectra: np.ndarray, methods: list, y: np.ndarray = None, groups: np.ndarray = None):
        self.fit(spectra, methods, y=y, groups=groups)
        return self.transform(spectra, methods)

    # -------------------------
    # Interval selection methods
    # -------------------------
    def select_intervals_grouped(
        self,
        X, y, groups,
        n_intervals=100,
        n_components=2,
        cv_folds=5,
        select_mode="threshold",
        threshold=0.0,
        top_k=None,
        return_scores=False,
        plot=True,
        stability_repeats=1,
        stability_keep=0.0,   # e.g., 0.6 keeps intervals selected in ≥60% of repeats
    ):

        """
        Grouped iPLS scoring that avoids leakage and scores R² on sample means.

        Parameters
        ----------
        X : array, shape (n_spectra, n_features)
            Unsupervised-preprocessed spectra (what the model will actually see).
        y : array, shape (n_spectra,)
            Target values, repeated for replicates.
        groups : array, shape (n_spectra,)
            Sample id per spectrum (identical id for replicates).
        n_intervals : int
            Number of contiguous intervals to split the feature axis into.
        n_components : int
            PLS components for per-interval models.
        cv_folds : int
            GroupKFold splits.
        select_mode : {"threshold", "topk"}
            How to choose intervals.
        threshold : float
            Keep intervals with mean grouped-CV R² >= threshold (if select_mode="threshold").
        top_k : int or None
            Keep top-K intervals by grouped-CV R² (if select_mode="topk").
        return_scores : bool
            If True, also return per-interval scores.
        plot : bool
            If True, produce and close a small diagnostic plot.

        Returns
        -------
        sel : 1D int array
            Concatenated feature indices of selected intervals.
        (optional) interval_scores : list of (indices_array, mean_r2)
        """
        n_spectra, n_features = X.shape
        intervals = np.array_split(np.arange(n_features, dtype=int), n_intervals)
        rng = np.random.RandomState(42)
        interval_vote = np.zeros(n_intervals, dtype=int)
        
        def _grouped_splits_once():
            # Build randomized group folds by shuffling unique group labels each repeat
            unique_groups = np.unique(groups)
            rng.shuffle(unique_groups)
            # assign groups to cv_folds in round-robin buckets
            buckets = {k: [] for k in range(cv_folds)}
            for i, g in enumerate(unique_groups):
                buckets[i % cv_folds].append(g)
            # yield splits
            for k in range(cv_folds):
                val_groups = np.array(buckets[k])
                val_mask = np.isin(groups, val_groups)
                train_idx = np.where(~val_mask)[0]
                val_idx   = np.where(val_mask)[0]
                yield train_idx, val_idx

        interval_scores = np.zeros(n_intervals, dtype=float)
        
        for rep in range(int(stability_repeats)):
            # score each interval this repeat
            rep_scores = []
            for j, inds in enumerate(intervals):
                Xi = X[:, inds]
                fold_scores = []
                for tr, vl in _grouped_splits_once():
                    pls = PLSRegression(n_components=n_components)
                    pls.fit(Xi[tr], y[tr])
                    yhat = pls.predict(Xi[vl]).ravel()
                    fold_scores.append(_r2_on_sample_means(y[vl], yhat, groups[vl]))
                rep_scores.append(float(np.mean(fold_scores)))
        
            rep_scores = np.asarray(rep_scores)
            interval_scores += rep_scores
        
            # selection in this repeat
            if select_mode == "threshold":
                chosen_idx = np.where(rep_scores >= threshold)[0]
            elif select_mode == "topk":
                k = int(top_k)
                chosen_idx = np.argsort(rep_scores)[::-1][:k]
            else:
                raise ValueError("select_mode must be 'threshold' or 'topk'")
            interval_vote[chosen_idx] += 1
        
        # average score across repeats
        interval_scores /= float(stability_repeats)
        
        # final selection by the main rule…
        if select_mode == "threshold":
            chosen_idx = np.where(interval_scores >= threshold)[0]
        elif select_mode == "topk":
            k = int(top_k)
            chosen_idx = np.argsort(interval_scores)[::-1][:k]
        
        # …and (optionally) filtered by stability frequency
        if stability_keep > 0.0 and stability_repeats > 1:
            keep_min = int(np.ceil(stability_keep * stability_repeats))
            stable_idx = np.where(interval_vote >= keep_min)[0]
            chosen_idx = np.intersect1d(chosen_idx, stable_idx, assume_unique=False)
        
        chosen = [intervals[i] for i in chosen_idx]


        # Selection policy
        if select_mode == "threshold":
            chosen = [inds for inds, score in interval_scores if score >= threshold]
        elif select_mode == "topk":
            if top_k is None:
                raise ValueError("select_mode='topk' requires top_k")
            ranked = sorted(interval_scores, key=lambda t: t[1], reverse=True)
            chosen = [inds for inds, _ in ranked[:int(top_k)]]
        else:
            raise ValueError("select_mode must be 'threshold' or 'topk'")

        num_selected = len(chosen)
        print(f"IntervalPLS(Grouped): selected {num_selected}/{n_intervals} intervals "
              f"(mode={select_mode}, thresh={threshold}, top_k={top_k})")

        if not chosen:
            sel = np.arange(n_features, dtype=int)
        else:
            sel = np.hstack(chosen).astype(int)

        # optional diagnostics (closed immediately; safe for Agg)
        if plot:
            scores_only = [s for _, s in interval_scores]
            fig, (ax_spec, ax_map) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            # rough context plot (use mean spec since X is already preprocessed)
            ax_spec.plot(np.arange(n_features), X.mean(axis=0), color='gray', alpha=0.8)
            for block in intervals:
                ax_spec.axvline(block[0], color='black', linestyle='--', linewidth=0.5)
            ax_spec.set_ylabel("Mean intensity (preprocessed)")
            ax_spec.set_title("Intervals over feature axis")

            cmap = mpl.cm.get_cmap('viridis')
            norm = mpl.colors.Normalize(vmin=min(scores_only), vmax=max(scores_only))
            for (inds, s) in interval_scores:
                ax_map.plot(inds, np.zeros_like(inds), color=cmap(norm(s)), linewidth=4)
            ax_map.set_xlabel("Feature index")
            ax_map.set_yticks([])
            ax_map.set_ylabel("Score bands")
            ax_map.set_title("Interval-wise grouped-CV R²")

            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array(scores_only)
            fig.colorbar(sm, ax=ax_map, label='Grouped-CV R²', orientation='vertical')
            plt.tight_layout()
            plt.close(fig)

        if return_scores:
            return sel, interval_scores
        return sel

    def select_intervals_by_pls_r2(self, X, y, n_intervals=150, n_components=2, cv_folds=5, threshold=0.0):
        """
        Legacy (non-grouped) iPLS scoring using KFold and R² on spectra-level predictions.
        Kept for compatibility; prefer `select_intervals_grouped`.
        Returns a 1D int array of selected feature indices.
        """
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

        num_selected = len(selected)
        print(f"IntervalPLS (legacy KFold): selected {num_selected}/{n_intervals} intervals (threshold={threshold})")

        # small diagnostic (closed immediately)
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
        ax_map.set_title("Interval-wise R² mapping (legacy)")

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(r2_scores)
        fig.colorbar(sm, ax=ax_map, label='Interval R²', orientation='vertical')
        plt.tight_layout()
        plt.close(fig)

        if not selected:
            return np.arange(n_features, dtype=int)
        return np.hstack(selected).astype(int)
