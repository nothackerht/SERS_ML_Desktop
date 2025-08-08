# modules/bayes_opt.py

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.utils import use_named_args
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from modules.preprocessing import Preprocessing

def optimize_xgb_with_cv(
    X_pool, y_pool, groups, chain, xgb_space, y_tr_meta, y_ex_meta,  # y_* kept for signature compatibility
    n_calls=25
):
    """
    Bayesian optimization with 5-fold GroupKFold CV on (X_pool, y_pool).
    Preprocessing is fold-safe: fit on inner-train, transform inner-val.
    IntervalPLS must be handled outside this function.
    """
    # Remove IntervalPLS at the inner-CV level (if present in chain)
    methods_wo_ipls = [m for m in chain if m != 'IntervalPLS']

    # Ensure proper shapes
    groups = np.asarray(groups).ravel()
    assert X_pool.shape[1] == len(groups) == len(y_pool), "X/y/groups length mismatch in BayesOpt CV."

    # Pick device safely
    try:
        import torch
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        _device = 'cpu'

    @use_named_args(xgb_space)
    def objective(**params):
        rmses = []
        gkf = GroupKFold(n_splits=5)

        for train_idx, val_idx in gkf.split(X_pool.T, y_pool, groups):
            # Shapes: (n_samples, n_features)
            X_train = X_pool[:, train_idx].T
            y_train = y_pool[train_idx]
            X_val   = X_pool[:, val_idx].T
            y_val   = y_pool[val_idx]
            g_val   = groups[val_idx]

            # Preprocessing (train → fit_transform, val → transform)
            fold_prep = Preprocessing(ipls_threshold=0.2)
            X_train_proc = fold_prep.fit_transform(X_train.T, methods_wo_ipls, y=y_train).T
            X_val_proc   = fold_prep.transform(X_val.T, methods_wo_ipls).T

            model = XGBRegressor(
                **params,
                tree_method='hist',
                device=_device,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                eval_metric='rmse',
            )
            model.fit(X_train_proc, y_train)

            preds = model.predict(X_val_proc)

            # Group-wise mean (9 spectra → 1 per sample)
            df = pd.DataFrame({'group': g_val, 'true': y_val, 'pred': preds})
            grouped = df.groupby('group', as_index=False).agg({'true': 'first', 'pred': 'mean'})

            rmse = float(np.sqrt(mean_squared_error(grouped['true'], grouped['pred'])))
            rmses.append(rmse)

        return float(np.mean(rmses))

    result = gp_minimize(
        objective,
        dimensions=xgb_space,
        n_calls=n_calls,
        random_state=42
    )
    return result
