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
from skopt.plots import plot_convergence, plot_evaluations

# Default output directory
OUTPUT_DIR = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\bayes_results"


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_convergence_curve(res):
    """
    Plot the optimization convergence (trial vs. CV RMSE).
    """
    _ensure_dir()
    ax = plot_convergence(res)
    fig = ax.figure
    fig.savefig(
        os.path.join(OUTPUT_DIR, 'convergence_curve.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


def plot_evaluations_scatter(res):
    """
    Plot pairwise scatter and marginal distributions of hyperparameters vs. performance.
    """
    _ensure_dir()
    ax = plot_evaluations(res)
    fig = ax.figure
    fig.savefig(
        os.path.join(OUTPUT_DIR, 'evaluations_plot.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


def plot_hyperparam_heatmap(res, dim1, dim2):
    """
    Scatter heatmap of two hyperparameters vs. CV RMSE.
    """
    _ensure_dir()
    # get axis indices for dim1, dim2
    names = [d.name for d in res.space.dimensions]
    try:
        i1 = names.index(dim1)
        i2 = names.index(dim2)
    except ValueError:
        print(f"Dimension names {dim1},{dim2} not in search space.")
        return
    xs = [x[i1] for x in res.x_iters]
    ys = [x[i2] for x in res.x_iters]
    zs = res.func_vals

    plt.figure(figsize=(6,5))
    sc = plt.scatter(xs, ys, c=zs, cmap='viridis', edgecolor='k')
    plt.colorbar(sc, label='CV RMSE')
    plt.xlabel(dim1)
    plt.ylabel(dim2)
    plt.title(f"Heatmap: {dim1} vs {dim2}")
    fname = f"heatmap_{dim1}_vs_{dim2}.png"
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, fname),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


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


def plot_parity(y_test, y_pred):
    """
    Parity plot (Predicted vs. Actual) with 45-degree line.
    """
    _ensure_dir()
    mn, mx = min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([mn, mx], [mn, mx], '--', color='gray')
    plt.xlabel("Actual target_SI")
    plt.ylabel("Predicted target_SI")
    plt.title("Parity Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'parity_plot.png'), dpi=300)
    plt.close()

# SHAP summary optional

def plot_shap_summary(model, X, feature_names=None):
    """
    SHAP summary plot for tree-based model, if shap is installed.
    """
    try:
        import shap
    except ImportError:
        print("shap not installed; skipping SHAP summary plot.")
        return
    _ensure_dir()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), dpi=300)
    plt.close()
