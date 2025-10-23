# make_pipeline_triplets_std.py
import os, glob, math, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── USER PATHS ─────────────────────────────────────────────────────────────────
PATH_CV   = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Matthews_code_Global\Results_with_controls\results_global_combined"
PATH_SPXY = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\spxy_eval\Boxes1-3 with controls no leak"
PATH_EXT  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\external_test_eval\With controls"

# Targets to plot (order = panels left→right)
TARGETS = [
    ("SI",           "Splicing Index (SI)",            "Actual Splicing Index",                 "Predicted Splicing Index"),
    ("HGS_pp_avg",   "Hand Grip Strength (%)",         "Actual Hand Grip Strength (%)",         "Predicted Hand Grip Strength (%)"),
    ("ADF_pp_avg",   "Average Ankle Dorsiflexion (%)", "Actual Average Ankle Dorsiflexion (%)", "Predicted Average Ankle Dorsiflexion (%)"),
]

# If auto-discovery misses your per-sample predictions, hard-wire them here:
FILE_OVERRIDES = {
    # "cv": {
    #     "SI": r"...\SI__winner_predictions.xlsx",
    #     "HGS_pp_avg": r"...\HGS_pp_avg__winner_predictions.xlsx",
    #     "ADF_pp_avg": r"...\ADF_pp_avg__winner_predictions.xlsx",
    # },
    # "spxy": {
    #     "SI": r"...\target_SI__SPXY_predictions.xlsx",
    #     "HGS_pp_avg": r"...\HGS_pp_avg__SPXY_predictions.xlsx",
    #     "ADF_pp_avg": r"...\ADF_pp_avg__SPXY_predictions.xlsx",
    # },
    # "external": {
    #     "SI": r"...\target_SI__external_predictions.xlsx",
    #     "HGS_pp_avg": r"...\HGS_pp_avg__external_predictions.xlsx",
    #     "ADF_pp_avg": r"...\ADF_pp_avg__external_predictions.xlsx",
    # },
}

# Error bars are raw std (no scaling)
ERR_MULTIPLIER = 1.0

# File patterns to search for per-sample predictions
GLOBS = [
    "target_*__*pred*.xlsx", "target_*__*pred*.csv",
    "*winner*pred*.xlsx", "*winner*pred*.csv",
    "*oof*pred*.xlsx", "*oof*pred*.csv",
    "*predictions*.xlsx", "*predictions*.csv",
    "*parity*.xlsx", "*parity*.csv",
    "*pred*.xlsx", "*pred*.csv",
]

# Column name keys
TRUE_KEYS = {"y_true","true","actual","target","observed","reference","ref","ground_truth"}
PRED_KEYS = {"y_pred","pred","yhat","estimate","prediction","fit"}
STD_KEYS  = {"y_pred_std","ystd","stderr","std","sd","sigma","pred_std"}

# ── I/O helpers ───────────────────────────────────────────────────────────────
def _read_any(path):
    if path.lower().endswith(".csv"):
        return {"__csv__": pd.read_csv(path)}
    return pd.read_excel(path, sheet_name=None)

def _pick_cols_for_target(df, target_key):
    """
    Find (y_true, y_pred, y_std) for `target_key`.
    Priority 1: explicit per-target columns, e.g.,
        target_SI__true / target_SI__pred / target_SI__pred_std
        HGS_pp_avg__true / HGS_pp_avg__pred / HGS_pp_avg__pred_std
        ADF_pp_avg__true / ...
    Priority 2: generic names (y_true/y_pred/y_pred_std)
    Fallback: first two numeric columns (no std).
    """
    cols = list(df.columns)
    # build prefix for explicit forms
    base = f"{'target_' if target_key=='SI' else ''}{target_key}__"
    y_true = next((c for c in cols if c.lower() == (base + "true").lower()), None)
    y_pred = next((c for c in cols if c.lower() == (base + "pred").lower()), None)
    y_std  = next((c for c in cols if c.lower() == (base + "pred_std").lower()), None)
    if y_true and y_pred:
        return y_true, y_pred, y_std

    # generic names
    lower = {c: c.lower() for c in cols}
    def find(keys):
        for c, lc in lower.items():
            if lc in keys or any(re.search(rf"\b{re.escape(k)}\b", lc) for k in keys):
                return c
        return None
    y_true = find(TRUE_KEYS)
    y_pred = find(PRED_KEYS)
    y_std  = find(STD_KEYS)
    if y_true and y_pred:
        return y_true, y_pred, y_std

    # fallback
    nums = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(nums) >= 2:
        return nums[0], nums[1], None
    return None, None, None

def _load_target_df(file_path, target_key):
    """Load a DataFrame with columns: y_true, y_pred, and (optional) y_std."""
    sheets = _read_any(file_path)
    for _, df in sheets.items():
        if not isinstance(df, pd.DataFrame):
            continue
        y_true, y_pred, y_std = _pick_cols_for_target(df, target_key)
        if y_true and y_pred:
            out = pd.DataFrame({
                "y_true": pd.to_numeric(df[y_true], errors="coerce"),
                "y_pred": pd.to_numeric(df[y_pred], errors="coerce"),
            }).dropna(subset=["y_true", "y_pred"])
            if y_std and y_std in df.columns:
                out["y_std"] = pd.to_numeric(df[y_std], errors="coerce") * ERR_MULTIPLIER
            return out
    raise FileNotFoundError(f"No usable columns for target '{target_key}' in {file_path}")

def _auto_find_file(root, target_key):
    """Search for a predictions file under <root>/<target_key> first, then <root>."""
    for base in [os.path.join(root, target_key), root]:
        if not os.path.isdir(base):
            continue
        # prefer files whose names mention the target
        for pat in GLOBS:
            for f in glob.glob(os.path.join(base, "**", pat), recursive=True):
                name = os.path.basename(f).lower()
                if target_key.lower() in name or f"target_{target_key}".lower() in name:
                    return f
        # fallback to first match of any pattern
        for pat in GLOBS:
            hits = glob.glob(os.path.join(base, "**", pat), recursive=True)
            if hits:
                return hits[0]
    return None

def _get_pipeline_data(pipeline, root):
    """
    Returns dict: {target_key: DataFrame(y_true, y_pred, [y_std])}
    Requires per-sample prediction files (not summary metrics).
    """
    data = {}
    for key, *_ in TARGETS:
        path = FILE_OVERRIDES.get(pipeline, {}).get(key)
        if not path:
            path = _auto_find_file(root, key)
        if not path:
            raise FileNotFoundError(f"[{pipeline}] No predictions file found for '{key}' under {root}")
        data[key] = _load_target_df(path, key)
    return data

# ── Plot helpers ──────────────────────────────────────────────────────────────
def _fit_line(x, y):
    m, b = np.polyfit(x, y, 1)
    yhat = m*x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    rmse_id = math.sqrt(np.mean((y - x)**2))  # RMSE vs identity
    return m, b, r2, rmse_id

def _parity(ax, df, title, xlab, ylab):
    x = df["y_true"].to_numpy(float)
    y = df["y_pred"].to_numpy(float)

    if "y_std" in df.columns:
        ax.errorbar(x, y, yerr=df["y_std"].to_numpy(float), fmt='o', ms=3, alpha=0.5, capsize=2)
    else:
        ax.plot(x, y, 'o', ms=4, alpha=0.95)

    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])
    pad = 0.05 * (hi - lo if hi > lo else 1.0)

    # identity and fit lines
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], '--', linewidth=1)
    m, b, r2, rmse = _fit_line(x, y)
    xx = np.linspace(lo-pad, hi+pad, 100)
    ax.plot(xx, m*xx + b, '-', linewidth=1)

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(lo-pad, hi+pad)
    ax.set_ylim(lo-pad, hi+pad)
    ax.set_aspect('equal', adjustable='box')

    ax.text(0.05, 0.95, f"R²={r2:.3f}\nSlope={m:.2f}\nIntercept={b:.2f}\nRMSE(id)={rmse:.3f}",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

def _triplet(title, data_map, outpng):
    # a bit taller to give space for the suptitle
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    # space for the suptitle; also add horizontal space between panels
    fig.subplots_adjust(top=0.80, wspace=0.28)

    for ax, (key, pretty, xlab, ylab) in zip(axes, TARGETS):
        if key in data_map:
            _parity(ax, data_map[key], pretty, xlab, ylab)
            # push panel titles down a touch
            ax.set_title(pretty, pad=12)
        else:
            ax.axis("off"); ax.set_title(f"{pretty}\n(no per-sample data)", pad=12)

    # put the pipeline title higher so it doesn't crowd panel titles
    fig.suptitle(title, y=0.96, fontsize=13)
    fig.savefig(outpng, dpi=300, bbox_inches="tight")
    print("Saved:", outpng)

# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("parity_triplets", exist_ok=True)

    # 1) 10-fold CV (Boxes 1–3)
    try:
        cv = _get_pipeline_data("cv", PATH_CV)
    except Exception as e:
        print("CV warning:", e); cv = {}
    _triplet("10-fold CV (Boxes 1–3)", cv, os.path.join("parity_triplets", "parity_cv_triplet.png"))

    # 2) SPXY (Boxes 1–3)
    try:
        spxy = _get_pipeline_data("spxy", PATH_SPXY)
    except Exception as e:
        print("SPXY warning:", e); spxy = {}
    _triplet("SPXY (Boxes 1–3)", spxy, os.path.join("parity_triplets", "parity_spxy_triplet.png"))

    # 3) External Test (Box 3)
    try:
        ext = _get_pipeline_data("external", PATH_EXT)
    except Exception as e:
        print("External warning:", e); ext = {}
    _triplet("External Test (Box 3)", ext, os.path.join("parity_triplets", "parity_external_triplet.png"))
