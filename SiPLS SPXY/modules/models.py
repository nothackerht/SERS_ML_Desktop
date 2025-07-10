# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 14:13:28 2025

@author: spect
"""

# modules/models.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import xgboost as xgb

from modules.torch_pls import TorchPLS  # Ensure this supports the required interface

class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class PLSModel(BaseModel):
    def __init__(self, n_components=10):
        self.model = TorchPLS(n_components=n_components, device="cuda")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class RFModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y.ravel())

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)

    def feature_importances(self):
        return self.model.feature_importances_


class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = xgb.XGBRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y.ravel())

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)

    def feature_importances(self):
        return self.model.feature_importances_


class SVRModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = SVR(kernel='rbf', **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y.ravel())

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)


class MLPModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = MLPRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y.ravel())

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)


class KNNModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = KNeighborsRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y.ravel())

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)


class GPRModel(BaseModel):
    def __init__(self):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    def fit(self, X, y):
        self.model.fit(X, y.ravel())

    def predict(self, X):
        return self.model.predict(X, return_std=False).reshape(-1, 1)


# Factory function to dynamically select model

def get_model(model_type='PLS', **kwargs):
    model_type = model_type.upper()
    if model_type == 'PLS':
        return PLSModel(**kwargs)
    elif model_type == 'RF':
        return RFModel(**kwargs)
    elif model_type == 'XGB':
        return XGBoostModel(**kwargs)
    elif model_type == 'SVR':
        return SVRModel(**kwargs)
    elif model_type == 'MLP':
        return MLPModel(**kwargs)
    elif model_type == 'KNN':
        return KNNModel(**kwargs)
    elif model_type == 'GPR':
        return GPRModel()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
