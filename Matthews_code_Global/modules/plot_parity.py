# modules/plot_parity.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def parity_plot_sample_level(
    y_true_sample: np.ndarray,
    y_pred_sample: np.ndarray,
    *,
    y_true_sample_std: np.ndarray | None = None,   # <-- optional x-error bars
    y_pred_sample_std: np.ndarray | None = None,   # <-- optional y-error bars
    target_names=("Splicing Index", "Hand Grip Strength (%)", "Average Ankle Dorsiflexion (%)"),
    out_dir: str | None = None,
    fname_prefix: str = "parity",
    dpi: int = 600,
    title_suffix: str = ""   # e.g., "PLS | EMSC+SNV | HP: {'n_components': 8}"
):
    """
    Make K parity plots at sample level. If std arrays are provided (shape=N×K),
    draw error bars for each sample using xerr/yerr from the 9 spectra per sample.
    """
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    assert y_true_sample.shape == y_pred_sample.shape
    N, K = y_true_sample.shape

    has_xerr = isinstance(y_true_sample_std, np.ndarray) and y_true_sample_std.shape == (N, K)
    has_yerr = isinstance(y_pred_sample_std, np.ndarray) and y_pred_sample_std.shape == (N, K)

    for j in range(K):
        yt = y_true_sample[:, j].reshape(-1, 1)
        yp = y_pred_sample[:, j].reshape(-1, 1)
        yt1, yp1 = yt.ravel(), yp.ravel()

        # Optional error bars per sample
        xerr = y_true_sample_std[:, j] if has_xerr else None
        yerr = y_pred_sample_std[:, j] if has_yerr else None

        # Fit line + metrics
        lr = LinearRegression().fit(yt, yp)
        slope = float(lr.coef_[0])
        intercept = float(lr.intercept_)
        r2 = float(r2_score(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt1, yp1)))

        # Figure
        plt.figure(figsize=(6, 6))
        if (xerr is not None) or (yerr is not None):
            plt.errorbar(yt1, yp1, xerr=xerr, yerr=yerr, fmt='o', alpha=0.85,
                         capsize=4, capthick=1, ecolor='gray', markersize=5)
        else:
            plt.scatter(yt1, yp1, alpha=0.85)

        x_min, x_max = float(np.min(yt1)), float(np.max(yt1))
        grid = np.linspace(x_min, x_max, 200).reshape(-1, 1)
        plt.plot(grid, lr.predict(grid), 'r--', lw=2)        # best-fit line
        plt.plot([x_min, x_max], [x_min, x_max], 'k:', lw=1) # 45° line

        # Labels
        tname = target_names[j] if j < len(target_names) else f"Target {j+1}"
        title = tname + (f" — {title_suffix}" if title_suffix else "")
        plt.title(title, fontweight="bold")
        plt.xlabel(f"Actual {tname}", fontweight="bold")
        plt.ylabel(f"Predicted {tname}", fontweight="bold")

        # Stats box (now includes RMSE)
        legend_txt = f"RMSE: {rmse:.3f}\nR²: {r2:.3f}\nSlope: {slope:.2f}\nIntercept: {intercept:.2f}"
        plt.legend([legend_txt], loc='lower right', frameon=True, fancybox=True,
                   framealpha=0.6, handlelength=0, handletextpad=0)

        plt.tight_layout()
        if out_dir:
            fp = os.path.join(out_dir, f"{fname_prefix}_{j+1}.png")
            plt.savefig(fp, dpi=dpi)
        plt.close()
