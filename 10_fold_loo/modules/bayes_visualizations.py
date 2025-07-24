#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
bayes_visualizations.py

Visualization utilities for Bayesian hyperparameter optimization of XGBoost models on SERS data.
Saves all plots under:
    C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\bayes_results
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_evaluations, plot_objective

# Default output directory
OUTPUT_DIR = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\bayes_results"


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_convergence_curve(res):
    """
    Plot the optimization convergence (trial vs. CV RMSE).
    """
    _ensure_dir()
    fig = plot_convergence(res)
    fig.savefig(os.path.join(OUTPUT_DIR, 'convergence_curve.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_evaluations_scatter(res):
    """
    Plot pairwise scatter and marginal distributions of hyperparameters vs. performance.
    """
    _ensure_dir()
    fig = plot_evaluations(res)
    fig.savefig(os.path.join(OUTPUT_DIR, 'evaluations_plot.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_hyperparam_heatmap(res, dim1, dim2, n_samples=100):
    """
    Plot a 2D contour of the objective surface for two hyperparameters.

    Parameters:
    - res: OptimizeResult from gp_minimize
    - dim1, dim2: names of two dimensions (strings)
    - n_samples: resolution per axis
    """
    _ensure_dir()
    fig = plot_objective(res, dimensions=[dim1, dim2], n_samples=n_samples)
    fname = f"heatmap_{dim1}_vs_{dim2}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_feature_importance(model, top_n=20):
    """
    Bar chart of the top_n feature importances by gain.
    Assumes model is a fitted XGBRegressor.
    """
    _ensure_dir()
    imp = model.get_booster().get_score(importance_type='gain')
    items = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, gains = zip(*items)
    y_pos = np.arange(len(features))

    plt.figure(figsize=(8, max(6, 0.3*len(features))))
    plt.barh(y_pos, gains[::-1])
    plt.yticks(y_pos, [f for f in features[::-1]])
    plt.xlabel("Gain")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300)
    plt.close()


def plot_residuals(model, X_test, y_test):
    """
    Histogram of residuals on the test set.
    """
    _ensure_dir()
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(8,6))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Count")
    plt.title("Residuals Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_histogram.png'), dpi=300)
    plt.close()


def plot_parity(model, X_test, y_test):
    """
    Parity plot (Predicted vs. Actual) with 45-degree line.
    """
    _ensure_dir()
    y_pred = model.predict(X_test)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([mn, mx], [mn, mx], '--', color='gray')
    plt.xlabel("Actual target_SI")
    plt.ylabel("Predicted target_SI")
    plt.title("Parity Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'parity_plot.png'), dpi=300)
    plt.close()


def plot_shap_summary(model, X, feature_names=None):
    """
    SHAP summary plot for tree-based model.
    Requires shap library.
    """
    _ensure_dir()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), dpi=300)
    plt.close()
