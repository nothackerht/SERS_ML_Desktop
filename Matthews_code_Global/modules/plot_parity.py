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
        # Optional: drop NaNs before metrics/plotting
        mask = ~np.isnan(yt1) & ~np.isnan(yp1)
        if xerr is not None:
            mask &= ~np.isnan(xerr)
        if yerr is not None:
            mask &= ~np.isnan(yerr)
        
        yt  = yt[mask].reshape(-1, 1)
        yp  = yp[mask].reshape(-1, 1)
        yt1 = yt1[mask]
        yp1 = yp1[mask]
        if xerr is not None:
            xerr = xerr[mask]
        if yerr is not None:
            yerr = yerr[mask]

        # Fit line + metrics
        lr = LinearRegression().fit(yt, yp)
        slope = float(np.ravel(lr.coef_)[0])     # <- flatten for safety
        intercept = float(lr.intercept_)
        r2 = float(r2_score(yt1, yp1))           # <- use 1-D arrays
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
        plt.axis('equal')  # optional but recommended

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
def save_outer_predictions_excel(
    y_true_sample: np.ndarray,
    y_pred_sample: np.ndarray,
    meta_kept: pd.DataFrame | None = None,
    target_order=("target_SI", "HGS_pp_avg", "ADF_pp_avg"),
    out_path: str | None = None,
    y_true_sample_std: np.ndarray | None = None,
    y_pred_sample_std: np.ndarray | None = None,
):
    """
    Save a tidy table of held-out *sample-level* predictions (and meta, if provided).
    Optionally includes per-sample std columns for error bars.

    Columns per target:
      <t>__true, <t>__pred, [optional] <t>__true_std, <t>__pred_std
    """
    assert y_true_sample.shape == y_pred_sample.shape, "true/pred shapes must match"
    N, K = y_true_sample.shape

    include_true_std = isinstance(y_true_sample_std, np.ndarray) and y_true_sample_std.shape == (N, K)
    include_pred_std = isinstance(y_pred_sample_std, np.ndarray) and y_pred_sample_std.shape == (N, K)

    data = {}
    for j, t in enumerate(target_order):
        data[f"{t}__true"] = y_true_sample[:, j]
        data[f"{t}__pred"] = y_pred_sample[:, j]
        if include_true_std:
            data[f"{t}__true_std"] = y_true_sample_std[:, j]
        if include_pred_std:
            data[f"{t}__pred_std"] = y_pred_sample_std[:, j]

    df = pd.DataFrame(data)
    if (meta_kept is not None) and (len(meta_kept) == N):
        df = pd.concat([meta_kept.reset_index(drop=True), df], axis=1)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_excel(out_path, index=False)
    return df
