# modules/plot_parity.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def parity_plot_sample_level(
    y_true_sample: np.ndarray,
    y_pred_sample: np.ndarray,
    target_names=("Splicing Index", "Hand Grip Strength (%)", "Average Ankle Dorsiflexion (%)"),
    out_dir: str = None,
    fname_prefix: str = "parity",
    dpi: int = 600,
    title_suffix: str = ""   # <— NEW: shows model/preproc in the title
):
    """
    Make 3 parity plots (sample-level). Each target uses averaged predictions/labels.
    """
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    assert y_true_sample.shape == y_pred_sample.shape
    K = y_true_sample.shape[1]

    for j in range(K):
        yt = y_true_sample[:, j].reshape(-1, 1)
        yp = y_pred_sample[:, j].reshape(-1, 1)

        # Fit line
        lr = LinearRegression().fit(yt, yp)
        slope = float(lr.coef_[0])
        intercept = float(lr.intercept_)
        r2 = float(r2_score(yt, yp))

        # Figure
        plt.figure(figsize=(6,6))
        plt.scatter(yt, yp, alpha=0.8)  # no edgecolor to avoid warning
        x_min, x_max = float(np.min(yt)), float(np.max(yt))
        grid = np.linspace(x_min, x_max, 200).reshape(-1,1)
        plt.plot(grid, lr.predict(grid), 'r--', lw=2)      # best-fit line
        plt.plot([x_min, x_max], [x_min, x_max], 'k:', lw=1)  # 45° line

        # Labels
        tname = target_names[j] if j < len(target_names) else f"Target {j+1}"
        title = tname + (f" — {title_suffix}" if title_suffix else "")
        plt.title(title, fontweight="bold")
        plt.xlabel(f"Actual {tname}", fontweight="bold")
        plt.ylabel(f"Predicted {tname}", fontweight="bold")

        # Stats box
        legend_txt = f"R²: {r2:.3f}\nSlope: {slope:.2f}\nIntercept: {intercept:.2f}"
        plt.legend([legend_txt], loc='lower right', frameon=True, fancybox=True,
                   framealpha=0.6, handlelength=0, handletextpad=0)

        plt.tight_layout()
        if out_dir:
            fp = os.path.join(out_dir, f"{fname_prefix}_{j+1}.png")
            plt.savefig(fp, dpi=dpi)
        plt.close()

def save_outer_predictions_excel(
    y_true_sample: np.ndarray,
    y_pred_sample: np.ndarray,
    meta_kept: pd.DataFrame = None,
    target_order=("target_SI", "HGS_pp_avg", "ADF_pp_avg"),
    out_path: str = None
):
    """
    Save a tidy table of held-out sample-level predictions (and meta if provided).
    """
    N, K = y_true_sample.shape
    data = {}
    for j, t in enumerate(target_order):
        data[f"{t}__true"] = y_true_sample[:, j]
        data[f"{t}__pred"] = y_pred_sample[:, j]

    df = pd.DataFrame(data)
    if meta_kept is not None and len(meta_kept) == N:
        df = pd.concat([meta_kept.reset_index(drop=True), df], axis=1)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_excel(out_path, index=False)
    return df
