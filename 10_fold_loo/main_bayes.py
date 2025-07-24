# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 12:32:08 2025

@author: spect
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_bayes.py

Performs Bayesian hyperparameter optimization for XGBoost on SERS spectral data.
"""
import os
import numpy as np
import torch
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

from modules.data_loader import load_data, load_metadata
from modules.bayes_opt import optimize_xgb
from modules.hyperparams import xgb_space

# Check GPU availability
def print_device_info():
    cuda_avail = torch.cuda.is_available()
    print("CUDA Available:", cuda_avail)
    if cuda_avail:
        print("Device:", torch.cuda.get_device_name(0))
    else:
        print("Device: CPU")

# ─────────────────────────────────────────────────────────────────────────────
# Paths (adjust as needed)
data_dir          = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
meta_path        = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Load SERS data
def load_training_data(data_dir, meta_path):
    wavenumbers, averaged_spectra, _, _ = load_data(
        data_dir,
        metadata_path=meta_path,
        return_filenames=True
    )
    y_df = load_metadata(meta_path)
    # averaged_spectra shape: (n_features, n_samples)
    X = averaged_spectra.T  # (n_samples, n_features)
    y = y_df['target_SI'].values
    return X, y

# ─────────────────────────────────────────────────────────────────────────────
# Objective function for Bayesian optimization
def xgb_objective(n_estimators,
                  learning_rate,
                  max_depth,
                  min_child_weight,
                  gamma,
                  subsample,
                  colsample_bytree,
                  reg_alpha,
                  reg_lambda):
    """
    Build an XGBRegressor with the given hyperparams,
    run 5-fold CV (negative MSE → RMSE), and return mean RMSE.
    """
    model = XGBRegressor(
        n_estimators     = n_estimators,
        learning_rate    = learning_rate,
        max_depth        = max_depth,
        min_child_weight = min_child_weight,
        gamma            = gamma,
        subsample        = subsample,
        colsample_bytree = colsample_bytree,
        reg_alpha        = reg_alpha,
        reg_lambda       = reg_lambda,
        tree_method      = 'hist',
        device           = 'cuda' if torch.cuda.is_available() else 'cpu',
        random_state     = 42,
    )
    # 5-fold CV on training data
    neg_mse = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    rmse = np.sqrt(-neg_mse)
    return float(rmse.mean())

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print_device_info()

    # Load data
    X_train, y_train = load_training_data(data_dir, meta_path)

    # Run Bayesian optimization
    best_params, best_loss = optimize_xgb(
        objective_fn = xgb_objective,
        search_space = xgb_space,
        n_calls      = 50,
        random_state = 42
    )

    print("🏆 Best hyperparameters:", best_params)
    print("📉 Best validation RMSE:", best_loss)

    # Retrain final model on full training set
    param_dict = dict(zip([dim.name for dim in xgb_space], best_params))
    final_model = XGBRegressor(
        **param_dict,
        tree_method = 'hist',
        device      = 'cuda' if torch.cuda.is_available() else 'cpu',
        random_state= 42
    )
    final_model.fit(X_train, y_train)

    # Save the trained model
    os.makedirs('models', exist_ok=True)
    import pickle
    with open('models/xgb_best.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print("✅ Final model saved to models/xgb_best.pkl")
