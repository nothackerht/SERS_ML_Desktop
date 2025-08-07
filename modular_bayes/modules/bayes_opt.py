# modules/bayes_opt.py

import numpy as np
from skopt import gp_minimize
from skopt.utils import use_named_args
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from modules.preprocessing import Preprocessing

def optimize_xgb_with_cv(
    X_pool, y_pool, groups, chain, xgb_space, y_tr_meta, y_ex_meta,
    n_calls=25
):
    """
    Performs Bayesian optimization with 5-fold GroupKFold CV using the provided X and y.
    IntervalPLS must be handled externally before calling this function.
    """
    methods_wo_ipls = [m for m in chain if m != 'IntervalPLS']

    @use_named_args(xgb_space)
    def objective(**params):
        rmses = []
        gkf = GroupKFold(n_splits=5)
    
        for train_idx, val_idx in gkf.split(X_pool.T, y_pool, groups):
            X_train = X_pool[:, train_idx].T
            y_train = y_pool[train_idx]
            X_val   = X_pool[:, val_idx].T
            y_val   = y_pool[val_idx]
            g_val   = groups[val_idx]
            print(f"[BayesOpt CV] Train: {X_train.shape}, Val: {X_val.shape}, Val Groups: {np.unique(g_val)}")

    
            # Preprocessing fit only on training fold
            fold_prep = Preprocessing(X_train.T, ipls_threshold=0.2)
            X_train_proc = fold_prep.preprocess(methods_wo_ipls, y=y_train)
            X_val_proc   = fold_prep.preprocess(methods_wo_ipls, y=y_val)
    
            model = XGBRegressor(
                **params,
                tree_method='hist',
                device='cuda',
                random_state=42
            )
            model.fit(X_train_proc, y_train)
            preds = model.predict(X_val_proc)
    
            # 🧠 Group predictions and average across spectra (9 per sample)
            import pandas as pd
            df = pd.DataFrame({'group': g_val, 'true': y_val, 'pred': preds})
            grouped = df.groupby('group').agg({'true': 'first', 'pred': 'mean'})
    
            rmse = np.sqrt(mean_squared_error(grouped['true'], grouped['pred']))
            rmses.append(rmse)
    
        return float(np.mean(rmses))


    result = gp_minimize(
        objective,
        dimensions=xgb_space,
        n_calls=n_calls,
        random_state=42
    )
    return result
