# modules/interval_selectors.py
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.cross_decomposition import PLSRegression
from dataclasses import dataclass

@dataclass
class SelectorConfig:
    n_components: int = 8
    n_intervals: int = 30         # number of fixed bins across wavenumbers
    max_intervals_keep: int = 6   # stopping cap for fiPLS/biPLS
    cv_splits: int = 5
    window_width: int = 60        # for mwPLS (points)
    window_step: int = 15         # for mwPLS (points)
    vip_threshold: float = 1.0    # for VIP-iPLS
    max_ga_intervals: int = 6     # for GA-iPLS (constraint)
    ga_pop: int = 30
    ga_gen: int = 25
    random_state: int = 13

def _group_cv_rmse(X, y, groups, n_components, cv_splits=5):
    gkf = GroupKFold(cv_splits)
    rmses = []
    for tr, va in gkf.split(X, y, groups):
        pls = PLSRegression(n_components=n_components)
        pls.fit(X[tr], y[tr])
        yhat = pls.predict(X[va]).ravel()
        rmses.append(np.sqrt(np.mean((y[va]-yhat)**2)))
    return float(np.mean(rmses))

def _make_equal_intervals(n_points, n_intervals):
    edges = np.linspace(0, n_points, n_intervals+1, dtype=int)
    return [(edges[i], edges[i+1]) for i in range(n_intervals)]

def _concat_intervals(X, intervals):
    idx = np.concatenate([np.arange(a,b) for a,b in intervals]) if intervals else np.array([], dtype=int)
    return idx
