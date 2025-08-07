# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 12:11:12 2025

@author: spect
"""

# modules/hyperparams.py

from skopt.space import Integer, Real

# XGBoost search space
xgb_space = [
    Integer(50,   500,   name="n_estimators"),
    Real   (1e-3, 0.3,   name="learning_rate"),
    Integer(2,    10,    name="max_depth"),
    Integer(1,    10,    name="min_child_weight"),
    Real   (0.0,  0.5,   name="gamma"),
    Real   (0.5,  1.0,   name="subsample"),
    Real   (0.5,  1.0,   name="colsample_bytree"),
    Real   (0.0,  1.0,   name="reg_alpha"),
    Real   (1.0,  5.0,   name="reg_lambda"),
]
