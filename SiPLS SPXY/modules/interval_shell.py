# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 14:23:28 2025

@author: spect
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
from modules.torch_pls import TorchPLS


def get_model_by_name(model_name, hyperparams):
    """
    Returns the model instance given its name and a dictionary of hyperparameters.
    """
    if model_name == 'sipls':
        return TorchPLS(
            n_components=hyperparams.get('n_components', 2,3,4,5,6),
            device=hyperparams.get('device', 'cpu')
        )
    elif model_name == 'random_forest':
        return RandomForestRegressor(
            n_estimators=hyperparams.get('n_estimators', 10),
            max_depth=hyperparams.get('max_depth', None),
            random_state=hyperparams.get('random_state', 42),
            n_jobs=-1
        )
    elif model_name == 'xgboost':
        return xgb.XGBRegressor(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', 6),
            learning_rate=hyperparams.get('learning_rate', 0.1),
            objective='reg:squarederror',
            random_state=hyperparams.get('random_state', 42),
            n_jobs=1  # ← avoid nested parallelism
        )

    elif model_name == 'svr':
        return SVR(
            C=hyperparams.get('C', 1.0),
            epsilon=hyperparams.get('epsilon', 0.1),
            kernel=hyperparams.get('kernel', 'rbf')
        )
    elif model_name == 'mlp':
        return MLPRegressor(
            hidden_layer_sizes=hyperparams.get('hidden_layer_sizes', (100,)),
            activation=hyperparams.get('activation', 'relu'),
            solver=hyperparams.get('solver', 'adam'),
            max_iter=hyperparams.get('max_iter', 1000),
            random_state=hyperparams.get('random_state', 42)
        )
    elif model_name == 'knn':
        return KNeighborsRegressor(
            n_neighbors=hyperparams.get('n_neighbors', 5),
            weights=hyperparams.get('weights', 'uniform'),
            algorithm=hyperparams.get('algorithm', 'auto')
        )
    elif model_name == 'gpr':
        return GaussianProcessRegressor(
            alpha=hyperparams.get('alpha', 1e-10),
            normalize_y=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")


# You can define a dictionary of default hyperparameters per model for grid search elsewhere
DEFAULT_HYPERPARAMETERS = {
    'sipls': {'n_components': 5, 'device': 'cpu'},
    'random_forest': {'n_estimators': 100, 'max_depth': None},
    'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
    'svr': {'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf'},
    'mlp': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000},
    'knn': {'n_neighbors': 5, 'weights': 'uniform'},
    'gpr': {'alpha': 1e-10}
}
