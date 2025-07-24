#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_bayes.py

Performs Bayesian hyperparameter optimization for XGBoost on SERS spectral data,
then generates and saves diagnostic visualizations.
"""
import os
import numpy as np
import torch
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split

from modules.data_loader import load_data, load_metadata
from modules.bayes_opt import optimize_xgb
from modules.hyperparams import xgb_space
from modules.bayes_visualizations import (
    plot_convergence_curve,
    plot_evaluations_scatter,
    plot_hyperparam_heatmap,
    plot_feature_importance,
    plot_residuals,
    plot_parity,
    plot_shap_summary
)

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
data_dir   = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
meta_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Load SERS data and split into train/test
def load_and_split(data_dir, meta_path, test_size=0.2, random_state=42):
    wv, avg_spec, _, _ = load_data(data_dir, metadata_path=meta_path, return_filenames=True)
    y_df = load_metadata(meta_path)
    X = avg_spec.T
    y = y_df['target_SI'].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

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
    Build an XGBRegressor with given hyperparams,
    run 5-fold CV on (X_train, y_train), and return mean RMSE.
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

    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split(data_dir, meta_path)
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Run Bayesian optimization and get full result
    res = optimize_xgb(
        objective_fn = xgb_objective,
        search_space = xgb_space,
        n_calls      = 50,
        random_state = 42
    )
    best_params, best_loss = res.x, res.fun

    print("🏆 Best hyperparameters:", best_params)
    print("📉 Best validation RMSE:", best_loss)

    # Retrain final model on full training set
    param_dict = dict(zip([dim.name for dim in xgb_space], best_params))
    final_model = XGBRegressor(
        **param_dict,
        tree_method  = 'hist',
        device       = 'cuda' if torch.cuda.is_available() else 'cpu',
        random_state = 42
    )
    final_model.fit(X_train, y_train)

    # Save the trained model
    os.makedirs('models', exist_ok=True)
    with open('models/xgb_best.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print("✅ Final model saved to models/xgb_best.pkl")

    # ── Generate and save visualizations
    plot_convergence_curve(res)
    plot_evaluations_scatter(res)
    # example heatmap for two dims
    plot_hyperparam_heatmap(res, 'max_depth', 'learning_rate')

    plot_feature_importance(final_model)
    plot_residuals(final_model, X_test, y_test)
    plot_parity(final_model, X_test, y_test)

    # Optional: SHAP summary on training set
    try:
        plot_shap_summary(final_model, X_train)
    except Exception as e:
        print("⚠️  SHAP summary plot failed:", e)
