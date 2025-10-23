# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 12:11:12 2025

@author: spect
"""

# modules/hyperparams.py

from skopt.space import Integer, Real

# XGBoost search space
xgb_space = [
    Integer(50,   1000,   name="n_estimators"),
    Real   (1e-4, 0.5,   name="learning_rate"),
    Integer(2,    16,   name="max_depth"),              # Support deeper trees
    Integer(1,    20,   name="min_child_weight"),       # More aggressive pruning
    Real   (0.0,  5.0,  name="gamma"),                  # Stronger regularization
    Real   (0.3,  1.0,  name="subsample"),              # Allow more dropout
    Real   (0.3,  1.0,  name="colsample_bytree"),       # Same for features
    Real   (0.0,  5.0,  name="reg_alpha"),              # L1 reg broader
    Real   (0.5,  10.0, name="reg_lambda"),             # L2 reg broader
]
