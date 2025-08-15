# -*- coding: utf-8 -*-
"""
Model factory for regression experiments.

Updated: accept **kwargs so callers can pass hyperparams directly, e.g.
get_model_by_name('sipls', n_components=2, device='cpu')
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb

from modules.torch_pls import TorchPLS


def get_model_by_name(model_name: str, **kwargs):
    """
    Return a model instance given its name and hyperparameters (as kwargs).

    Examples
    --------
    get_model_by_name('sipls', n_components=2, device='cpu')
    get_model_by_name('random_forest', n_estimators=200, max_depth=20)
    """
    name = model_name.lower()

    if name == 'sipls':
        # Accept common aliases and provide sensible defaults
        n_comp = (
            kwargs.pop('n_components',  # preferred
                       kwargs.pop('n_comp', 2))  # alias
        )
        device = kwargs.pop('device', 'cpu')
        return TorchPLS(n_components=n_comp, device=device, **kwargs)

    elif name == 'random_forest':
        return RandomForestRegressor(
            n_estimators=kwargs.pop('n_estimators', 100),
            max_depth=kwargs.pop('max_depth', None),
            random_state=kwargs.pop('random_state', 42),
            n_jobs=kwargs.pop('n_jobs', -1),
            **kwargs
        )

    elif name == 'xgboost':
        # Keep GPU-friendly defaults; allow overrides
        return xgb.XGBRegressor(
            n_estimators=kwargs.pop('n_estimators', 100),
            max_depth=kwargs.pop('max_depth', 6),
            learning_rate=kwargs.pop('learning_rate', 0.1),
            objective=kwargs.pop('objective', 'reg:squarederror'),
            random_state=kwargs.pop('random_state', 42),
            n_jobs=kwargs.pop('n_jobs', 1),            # avoid nested threading with joblib
            tree_method=kwargs.pop('tree_method', 'gpu_hist'),
            predictor=kwargs.pop('predictor', 'gpu_predictor'),
            **kwargs
        )

    elif name == 'svr':
        return SVR(
            C=kwargs.pop('C', 1.0),
            epsilon=kwargs.pop('epsilon', 0.1),
            kernel=kwargs.pop('kernel', 'rbf'),
            **kwargs
        )

    elif name == 'mlp':
        return MLPRegressor(
            hidden_layer_sizes=kwargs.pop('hidden_layer_sizes', (100,)),
            activation=kwargs.pop('activation', 'relu'),
            solver=kwargs.pop('solver', 'adam'),
            max_iter=kwargs.pop('max_iter', 1000),
            random_state=kwargs.pop('random_state', 42),
            **kwargs
        )

    elif name == 'knn':
        return KNeighborsRegressor(
            n_neighbors=kwargs.pop('n_neighbors', 5),
            weights=kwargs.pop('weights', 'uniform'),
            algorithm=kwargs.pop('algorithm', 'auto'),
            **kwargs
        )

    elif name == 'gpr':
        return GaussianProcessRegressor(
            alpha=kwargs.pop('alpha', 1e-10),
            normalize_y=kwargs.pop('normalize_y', True),
            **kwargs
        )

    else:
        raise ValueError(f"Unknown model type: {model_name}")


# Defaults you can reuse when constructing grids elsewhere
DEFAULT_HYPERPARAMETERS = {
    'sipls': {'n_components': 5, 'device': 'cpu'},
    'random_forest': {'n_estimators': 100, 'max_depth': None},
    'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
    'svr': {'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf'},
    'mlp': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000},
    'knn': {'n_neighbors': 5, 'weights': 'uniform'},
    'gpr': {'alpha': 1e-10}
}
