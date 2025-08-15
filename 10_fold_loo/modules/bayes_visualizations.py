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

# Default output directory (can be overridden via set_output_dir)
OUTPUT_DIR = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\bayes_results"


def set_output_dir(path: str) -> None:
    """Public helper to route plots to a specific directory."""
    global OUTPUT_DIR
    OUTPUT_DIR = path
    _ensure_dir()


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_convergence_curve(res):
    """Plot the optimization convergence (trial vs. CV RMSE)."""
    _ensure_dir()
    ax = plot_convergence(res)
    fig = ax.figure
    fig.savefig(os.path.join(OUTPUT_DIR, 'convergence_curve.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_evaluations_scatter(res):
    """Plot pairwise scatter and marginal distributions of hyperparameters vs. performance."""
    _ensure_dir()
    ax = plot_evaluations(res)
    fig = ax.figure
    fig.savefig(os.path.join(OUTPUT_DIR, 'evaluations_plot.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_hyperparam_heatmap(res, dim1, dim2):
    """Scatter heatmap of two hyperparameters vs. CV RMSE."""
    _ensure_dir()
    names = [d.name for d in res.space.dimensions]
    try:
        i1 = names.index(dim1)
        i2 = names.index(dim2)
    except ValueError:
        print(f"[viz] Dimension names {dim1},{dim2} not in search space.")
        return
    xs = [x[i1] for x in res.x_iters]
    ys = [x[i2] for x in res.x_iters]
    zs = res.func_vals

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(xs, ys, c=zs, cmap='viridis', edgecolor='k')
    plt.colorbar(sc, label='CV RMSE')
    plt.xlabel(dim1)
    plt.ylabel(dim2)
    plt.title(f"Heatmap: {dim1} vs {dim2}")
    fname = f"heatmap_{dim1}_vs_{dim2}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, top_n=20):
    """Bar chart of the top_n feature importances by gain. Assumes model is a fitted XGBRegressor."""
    _ensure_dir()
    try:
        imp = model.get_booster().get_score(importance_type='gain')
    except Exception as e:
        print(f"[viz] Could not extract feature importances: {e}")
        return
    if not imp:
        print("[viz] No feature importances available (empty booster score). Skipping plot.")
        return

    items = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, gains = zip(*items)
    y_pos = np.arange(len(features))

    plt.figure(figsize=(8, max(6, 0.3 * len(features))))
    plt.barh(y_pos, list(gains)[::-1])
    plt.yticks(y_pos, [f for f in list(features)[::-1]])
    plt.xlabel("Gain")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300)
    plt.close()


def plot_residuals(model, X_test, y_test):
    """Histogram of residuals on the given set."""
    _ensure_dir()
    y_pred = model.predict(X_test)
    residuals = np.asarray(y_test) - np.asarray(y_pred)

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Count")
    plt.title("Residuals Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_histogram.png'), dpi=300)
    plt.close()


def plot_parity(model_name, y_test, y_pred, stds=None, show_r2=True):
    """
    Parity plot (Predicted vs. Actual) with 45-degree line.
    If stds is provided, uses vertical error bars.
    Saves into a subfolder named after 'model_name' under OUTPUT_DIR.
    """
    _ensure_dir()

    # Coerce to 1D numpy arrays and drop NaNs if present
    y_test = np.asarray(y_test, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if stds is not None:
        stds = np.asarray(stds, dtype=float).ravel()
        if stds.shape != y_pred.shape:
            print("[viz] stds length does not match y_pred; ignoring stds.")
            stds = None

    # Create subfolder per model/prep name
    safe_dir = (model_name or "parity").replace(" ", "_").replace("/", "-")
    plot_dir = os.path.join(OUTPUT_DIR, safe_dir)
    os.makedirs(plot_dir, exist_ok=True)

    # Robust min/max (ignore NaNs)
    mn = np.nanmin([np.nanmin(y_test), np.nanmin(y_pred)])
    mx = np.nanmax([np.nanmax(y_test), np.nanmax(y_pred)])
    if not np.isfinite(mn) or not np.isfinite(mx):
        print("[viz] Non-finite data for parity plot; skipping.")
        return
    if mn == mx:
        # Avoid zero-span plots
        mn -= 1e-6
        mx += 1e-6

    plt.figure(figsize=(6, 6))
    if stds is not None:
        plt.errorbar(y_test, y_pred, yerr=stds, fmt='o', capsize=4, alpha=0.8)
    else:
        plt.scatter(y_test, y_pred, alpha=0.8)

    plt.plot([mn, mx], [mn, mx], '--', color='gray')
    plt.xlabel("Actual target_SI")
    plt.ylabel("Predicted target_SI")

    title = "Parity Plot"
    if model_name:
        title += f": {model_name}"

    if show_r2:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        try:
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            title += f"\nR² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}"
        except Exception as e:
            print(f"[viz] Metrics failed: {e}")

    plt.title(title)
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    filename = f"parity_plot_{safe_dir}.png"
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()


def plot_shap_summary(model, X, feature_names=None):
    """SHAP summary plot for tree-based model, if shap is installed."""
    try:
        import shap
    except ImportError:
        print("shap not installed; skipping SHAP summary plot.")
        return
    _ensure_dir()
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"[viz] SHAP plotting failed: {e}")
