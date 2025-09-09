# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 15:19:32 2025

@author: spect
"""

# -*- coding: utf-8 -*-
"""
Diagnostics Suite:
- Domain shift (PCA/UMAP) + nearest-neighbor distances (train vs external)
- Small calibration transfer (slope/intercept) using a few external standards
- Stability scans: SPXY repeats and grouped MCCV (flags "hard" samples)
- Label audit (train vs external distributions)

Place this file in modules/ and run from the project root:
    python -m modules.diagnostics_suite

If you run it from inside modules/, the path shim below will still work.
"""

import os, sys, re, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- ensure project root on sys.path if running from modules/ ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- optional UMAP (handled gracefully if missing) ---
try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

# --- your modules ---
from modules.data_loader import load_data
from modules.preprocessing import Preprocessing
from modules.plot_parity import parity_plot_sample_level, save_outer_predictions_excel
from modules.ten_fold_cv_regression_global import _best_interval_grouped

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, median_absolute_error,
    explained_variance_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# ---------------------- CONFIG ----------------------
TRAIN_DATA_DIR  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\data"
TRAIN_META_PATH = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata.csv"

EXT_DATA_DIR    = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\data_test_updated"
EXT_META_PATH   = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Data\y_metadata_test_updated_in_order.csv"

GLOBAL_RESULTS_DIR = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Matthews_code_Global\results_global_hp_per_target"

OUT_DIR = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = [
    ("Splicing Index", "target_SI"),
    ("Hand Grip Strength (%)", "HGS_pp_avg"),
    ("Average Ankle Dorsiflexion (%)", "ADF_pp_avg"),
]

REPS = 9          # spectra per sample
SPXY_CAL_FRAC = 0.80
SPXY_ALPHA    = 0.5
SPXY_REPEATS  = 20   # increase if you want more precision

MCCV_VAL_FRAC = 0.25
MCCV_REPEATS  = 200  # grouped MCCV iterations

# how many external samples to use for calibration transfer per draw
CAL_SAMPLES_PER_DRAW = 3
CAL_DRAWS            = 200

RANDOM_STATE = 42

# ---------------------- UTILS ----------------------
def _groups_for_spectra(n_samples: int, reps: int = 9) -> np.ndarray:
    return np.repeat(np.arange(n_samples, dtype=int), reps)

def _sample_means_stds(y_like: np.ndarray, reps: int = 9):
    y = np.asarray(y_like)
    assert (y.shape[0] % reps) == 0
    N = y.shape[0] // reps
    if y.ndim == 1:
        y3 = y.reshape(N, reps, 1)
        m  = y3.mean(axis=1).ravel()
        s  = y3.std(axis=1).ravel()
        return m, s
    y3 = y.reshape(N, reps, -1)
    return y3.mean(axis=1), y3.std(axis=1)

def _parse_methods(label: str) -> list[str]:
    if not label or label.lower().startswith("no"):
        return []
    return [p.strip() for p in label.split("+")]

def _parse_hp(model: str, hp_str: str):
    model = model.lower()
    if model == "pls":
        m = re.match(r"C=(\d+)", hp_str.strip());  assert m, f"Bad PLS hp: {hp_str}"
        return {"n_components": int(m.group(1))}
    if model == "ipls":
        m = re.match(r"C=(\d+)__I=(\d+)", hp_str.strip());  assert m, f"Bad iPLS hp: {hp_str}"
        return {"n_components": int(m.group(1)), "num_intervals": int(m.group(2))}
    if model == "rf":
        patt = r"ne(\d+)_md(\d+)_mss(\d+)_msl(\d+)_mf(\w+)"
        m = re.match(patt, hp_str.strip());        assert m, f"Bad RF hp: {hp_str}"
        ne, md, mss, msl, mf = m.groups()
        return {"n_estimators": int(ne), "max_depth": int(md),
                "min_samples_split": int(mss), "min_samples_leaf": int(msl),
                "max_features": "sqrt" if mf.lower()=="sqrt" else mf}
    if model == "acfnn":
        d = {}
        m = re.search(r"hls([0-9\-]+)", hp_str);   d["hidden_layer_sizes"] = tuple(int(x) for x in m.group(1).split("-")) if m else (128,64)
        m = re.search(r"act([a-zA-Z]+)", hp_str);  d["activation"] = (m.group(1) if m else "relu")
        m = re.search(r"bs(\d+)", hp_str);         d["batch_size"] = int(m.group(1)) if m else 64
        m = re.search(r"lr([0-9.]+)", hp_str);     d["learning_rate_init"] = float(m.group(1)) if m else 1e-3
        m = re.search(r"wd([0-9.]+)", hp_str);     d["alpha"] = float(m.group(1)) if m else 1e-4
        m = re.search(r"mi(\d+)", hp_str);         d["max_iter"] = int(m.group(1)) if m else 300
        return d
    raise ValueError(f"Unknown model: {model}")

def _fit_transform_pair(X_tr: np.ndarray, X_ap: np.ndarray, methods: list[str]):
    # scikit rows = samples; your Preprocessing expects (features, n)
    A = X_tr.T.copy(); B = X_ap.T.copy()
    for m in (methods or []):
        if m == "EMSC":
            ref = np.mean(A, axis=1)
            A = Preprocessing(A).emsc(A, reference=ref)
            B = Preprocessing(B).emsc(B, reference=ref)
        elif m == "Normalization":
            A = Preprocessing(A).normalize_spectrum(A)
            B = Preprocessing(B).normalize_spectrum(B)
        elif m == "SNV":
            A = Preprocessing(A).snv(A)
            B = Preprocessing(B).snv(B)
        elif m == "Second Derivative":
            A = Preprocessing(A).second_derivative(A)
            B = Preprocessing(B).second_derivative(B)
        elif m in ("No Preprocessing","No preprocessing","no preprocessing",""):
            pass
        else:
            raise ValueError(f"Unknown preprocessing: {m}")
    return A.T.astype(np.float32), B.T.astype(np.float32)

def _train_predict(model_name, methods, hp, Xtr, ytr, Xte, groups_tr):
    model_name = model_name.lower()
    if model_name == "pls":
        Xt_tr, Xt_te = _fit_transform_pair(Xtr, Xte, methods)
        mdl = PLSRegression(n_components=int(hp["n_components"]))
        mdl.fit(Xt_tr, ytr)
        return mdl.predict(Xt_te).ravel()
    if model_name == "rf":
        Xt_tr, Xt_te = _fit_transform_pair(Xtr, Xte, methods)
        rf = RandomForestRegressor(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            min_samples_split=hp["min_samples_split"], min_samples_leaf=hp["min_samples_leaf"],
            max_features=hp["max_features"], random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(Xt_tr, ytr.ravel())
        return rf.predict(Xt_te)
    if model_name == "acfnn":
        Xt_tr, Xt_te = _fit_transform_pair(Xtr, Xte, methods)
        mlp = MLPRegressor(random_state=RANDOM_STATE, **hp)
        mlp.fit(Xt_tr, ytr.ravel())
        return mlp.predict(Xt_te)
    if model_name == "ipls":
        num_intervals = int(hp["num_intervals"]); n_components = int(hp["n_components"])
        (a,b), _ = _best_interval_grouped(
            X_tr=Xtr, Y_tr=ytr, groups_tr=groups_tr,
            preprocess_methods=methods, n_components=n_components, num_intervals=num_intervals,
            reps=REPS, n_splits=max(3, min(5, int(len(np.unique(groups_tr))))),
            random_state=RANDOM_STATE
        )
        Xt_tr, Xt_te = _fit_transform_pair(Xtr[:, a:b], Xte[:, a:b], methods)
        mdl = PLSRegression(n_components=n_components)
        mdl.fit(Xt_tr, ytr)
        return mdl.predict(Xt_te).ravel()
    raise ValueError(f"Unsupported model: {model_name}")

def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "evs": float(explained_variance_score(y_true, y_pred)),
    }

def _spxy_split(Xsamp, ysamp, cal_frac=0.8, alpha=0.5, seed=None):
    rng = np.random.default_rng(seed)
    N = Xsamp.shape[0]
    Ncal = max(2, int(np.floor(cal_frac * N)))

    from sklearn.preprocessing import StandardScaler
    Xz = StandardScaler().fit_transform(Xsamp)
    yz = StandardScaler().fit_transform(ysamp.reshape(-1,1)).ravel()

    DX = np.sqrt(((Xz[:,None,:] - Xz[None,:,:])**2).sum(axis=2))
    DY = np.abs(yz[:,None] - yz[None,:])
    DX = DX / (DX.max() + 1e-12)
    DY = DY / (DY.max() + 1e-12)
    D  = alpha*DX + (1-alpha)*DY

    i0, j0 = np.unravel_index(np.argmax(D), D.shape)
    if i0 == j0:
        i0 = rng.integers(0, N)
        j0 = (i0 + rng.integers(1, N)) % N

    selected = [i0, j0]
    remaining = set(range(N)) - set(selected)

    while len(selected) < Ncal:
        mins = [(r, np.min(D[r, selected])) for r in remaining]
        r_star = max(mins, key=lambda t: t[1])[0]
        selected.append(r_star); remaining.remove(r_star)

    cal_idx = np.array(sorted(selected), dtype=int)
    val_idx = np.array(sorted(list(remaining)), dtype=int)
    return cal_idx, val_idx

# ---------------------- CORE ----------------------
def load_train_external():
    wn_tr, avg_tr, all_tr, meta_tr = load_data(
        data_dir=TRAIN_DATA_DIR, metadata_path=TRAIN_META_PATH,
        include_types=("DM1", "Control"), return_filenames=False,
        strict=False, report_samples=0,
    )
    wn_ex, avg_ex, all_ex, meta_ex = load_data(
        data_dir=EXT_DATA_DIR, metadata_path=EXT_META_PATH,
        include_types=("DM1","Control"), return_filenames=False,
        strict=False, report_samples=0,
    )
    return wn_tr, avg_tr, all_tr, meta_tr, wn_ex, avg_ex, all_ex, meta_ex

def winner_for_target(tcol):
    xlsx = os.path.join(GLOBAL_RESULTS_DIR, tcol, f"{tcol}__metrics_all_combos.xlsx")
    dfm = pd.read_excel(xlsx)
    if dfm is None or dfm.empty:
        raise FileNotFoundError(f"No rows in metrics file: {xlsx}")
    row = dfm.sort_values("rmse", ascending=True).iloc[0]
    model   = str(row["model"]).lower()
    methods = _parse_methods(str(row["preprocessing"]))
    hp_str  = str(row["hp"])
    hp      = _parse_hp(model, hp_str)
    return model, methods, hp, hp_str

def domain_shift_report():
    print("\n[1/4] Domain shift: PCA/UMAP + NN distances")
    wn_tr, avg_tr, all_tr, meta_tr, wn_ex, avg_ex, all_ex, meta_ex = load_train_external()

    # choose a neutral preprocessing chain for spectra visualization
    METHODS_FOR_SHIFT = ["EMSC", "Second Derivative"]

    Xtr = avg_tr.T     # (N_tr, 1731) averaged spectra
    Xex = avg_ex.T     # (N_ex, 1731)

    Xtr_prep, Xex_prep = _fit_transform_pair(Xtr, Xex, METHODS_FOR_SHIFT)
    scaler = StandardScaler().fit(Xtr_prep)
    Ztr = scaler.transform(Xtr_prep);  Zex = scaler.transform(Xex_prep)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    Ttr = pca.fit_transform(Ztr)
    Tex = pca.transform(Zex)

    out_dir = os.path.join(OUT_DIR, "shift")
    os.makedirs(out_dir, exist_ok=True)

    def _scatter(ax, A, B, title):
        ax.scatter(A[:,0], A[:,1], s=40, alpha=0.8, label="Train")
        ax.scatter(B[:,0], B[:,1], s=60, alpha=0.8, marker="^", label="External")
        ax.axhline(0, lw=0.2, c="k"); ax.axvline(0, lw=0.2, c="k")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(title); ax.legend()

    fig, ax = plt.subplots(1,1, figsize=(6,5), dpi=140)
    _scatter(ax, Ttr, Tex, f"PCA (Train vs External) — {METHODS_FOR_SHIFT}")
    evr = pca.explained_variance_ratio_
    ax.set_title(f"PCA (Train vs External) — {METHODS_FOR_SHIFT} | EVR: PC1={evr[0]:.2f}, PC2={evr[1]:.2f}")

    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pca_train_vs_external.png"), dpi=300); plt.close(fig)

    # Optional UMAP
    if HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
        U = reducer.fit_transform(np.vstack([Ztr, Zex]))
        Utr, Uex = U[:Ztr.shape[0]], U[Ztr.shape[0]:]
        fig, ax = plt.subplots(1,1, figsize=(6,5), dpi=140)
        _scatter(ax, Utr, Uex, f"UMAP (Train vs External) — {METHODS_FOR_SHIFT}")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "umap_train_vs_external.png"), dpi=300); plt.close(fig)
    else:
        print("  (UMAP not installed; skipping)")

    # Nearest-neighbor distances (in Z-space and in PCA space)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(5, Ztr.shape[0]), metric="euclidean").fit(Ztr)
    dists_z, _ = nn.kneighbors(Zex, return_distance=True)
    nn2 = NearestNeighbors(n_neighbors=min(5, Ttr.shape[0]), metric="euclidean").fit(Ttr)
    dists_pca, _ = nn2.kneighbors(Tex, return_distance=True)

    df = pd.DataFrame({
        "external_index": np.arange(Zex.shape[0]),
        "nn1_dist_Z": dists_z[:,0],
        "nn5_mean_dist_Z": dists_z.mean(axis=1),
        "nn1_dist_PCA": dists_pca[:,0],
        "nn5_mean_dist_PCA": dists_pca.mean(axis=1),
    })
    df.to_csv(os.path.join(out_dir, "external_nn_distances.csv"), index=False)
    # --- Quick NN-distance plots (Z-space and PCA-space) ---
    bins = max(10, int(np.sqrt(Zex.shape[0])))

    # Z-space: nearest (1-NN) distance histogram
    plt.figure(figsize=(6,4), dpi=140)
    plt.hist(dists_z[:, 0], bins=bins, alpha=0.85)
    plt.xlabel("Nearest neighbor distance (Z-space)")
    plt.ylabel("Count")
    plt.title("External → Train: 1-NN distance (Z-space)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nn1_dist_Z_hist.png"), dpi=300)
    plt.close()

    # Z-space: mean of k-NN (k=5) distances histogram
    plt.figure(figsize=(6,4), dpi=140)
    plt.hist(dists_z.mean(axis=1), bins=bins, alpha=0.85)
    plt.xlabel("Mean 5-NN distance (Z-space)")
    plt.ylabel("Count")
    plt.title("External → Train: mean 5-NN distance (Z-space)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nn5_mean_dist_Z_hist.png"), dpi=300)
    plt.close()

    # PCA-space (optional but handy for consistency with the PCA figure)
    plt.figure(figsize=(6,4), dpi=140)
    plt.hist(dists_pca[:, 0], bins=bins, alpha=0.85)
    plt.xlabel("Nearest neighbor distance (PCA space)")
    plt.ylabel("Count")
    plt.title("External → Train: 1-NN distance (PCA space)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nn1_dist_PCA_hist.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6,4), dpi=140)
    plt.hist(dists_pca.mean(axis=1), bins=bins, alpha=0.85)
    plt.xlabel("Mean 5-NN distance (PCA space)")
    plt.ylabel("Count")
    plt.title("External → Train: mean 5-NN distance (PCA space)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nn5_mean_dist_PCA_hist.png"), dpi=300)
    plt.close()

    # (Optional) save a few quantiles for quick thresholding
    qz = np.quantile(dists_z[:, 0], [0.5, 0.9, 0.95, 0.99]).tolist()
    qp = np.quantile(dists_pca[:, 0], [0.5, 0.9, 0.95, 0.99]).tolist()
    pd.DataFrame({
        "quantile": ["p50","p90","p95","p99"],
        "nn1_Z": qz,
        "nn1_PCA": qp
    }).to_csv(os.path.join(out_dir, "nn1_distance_quantiles.csv"), index=False)

    print(f"  Wrote PCA/UMAP plots + NN distance CSV → {out_dir}")

def predict_external_by_winner():
    """Train on full TRAIN, predict EXTERNAL, per target. Returns dict of predictions."""
    _, _, all_tr, meta_tr, _, _, all_ex, meta_ex = load_train_external()
    Xtr_all = all_tr.T
    Xex_all = all_ex.T
    Ntr = meta_tr.shape[0]; Nex = meta_ex.shape[0]

    preds = {}
    for nice, tcol in TARGETS:
        keep_tr = pd.to_numeric(meta_tr[tcol], errors="coerce").notna().to_numpy()
        ytr_s   = meta_tr.loc[keep_tr, tcol].to_numpy(float)
        keep_ex = pd.to_numeric(meta_ex[tcol], errors="coerce").notna().to_numpy()
        yex_s   = meta_ex.loc[keep_ex, tcol].to_numpy(float)

        model, methods, hp, hp_str = winner_for_target(tcol)

        # build spectra rows
        tr_idx = np.where(keep_tr)[0]; ex_idx = np.where(keep_ex)[0]
        row_tr = (tr_idx[:,None]*REPS + np.arange(REPS)).ravel()
        row_ex = (ex_idx[:,None]*REPS + np.arange(REPS)).ravel()

        Xtr = Xtr_all[row_tr]; Xex = Xex_all[row_ex]
        ytr = np.repeat(ytr_s.reshape(-1,1), REPS, axis=0)

        groups_tr = _groups_for_spectra(len(tr_idx), reps=REPS)
        yhat_spec = _train_predict(model, methods, hp, Xtr, ytr, Xex, groups_tr)
        yhat_ex, yhat_sd = _sample_means_stds(yhat_spec, reps=REPS)

        preds[tcol] = {
            "y_true_external": yex_s, "y_pred_external": yhat_ex, "y_pred_sd": yhat_sd,
            "keep_mask_external": keep_ex, "methods": methods, "model": model, "hp_str": hp_str
        }
    return preds

def calibration_transfer_simulation():
    print("\n[2/4] Calibration transfer: slope/intercept using few external samples")
    preds = predict_external_by_winner()
    out_dir = os.path.join(OUT_DIR, "calibration_transfer")
    os.makedirs(out_dir, exist_ok=True)

    draw_cache = {}  # target -> list of dicts with (repeat, a, b, cal_ids, test_ids, rmse_pre, rmse_post)
    rng = np.random.default_rng(RANDOM_STATE)

    for nice, tcol in TARGETS:
        # --- pull external truth/preds for this target ---
        y_true = preds[tcol]["y_true_external"].astype(float)
        y_pred = preds[tcol]["y_pred_external"].astype(float)
        y_sd   = preds[tcol]["y_pred_sd"].astype(float)
        Nex = len(y_true)

        if Nex < CAL_SAMPLES_PER_DRAW + 2:
            print(f"  [{tcol}] Too few external samples for calibration; skipping.")
            continue

        # ---- run many draws; keep per-target rows separate ----
        rows_t = []
        for r in range(CAL_DRAWS):
            cal_ids  = rng.choice(Nex, size=CAL_SAMPLES_PER_DRAW, replace=False)
            test_ids = np.setdiff1d(np.arange(Nex), cal_ids)

            # fit y = a + b * ŷ on calibration subset
            Xc   = np.c_[np.ones(cal_ids.size), y_pred[cal_ids]]
            beta = np.linalg.lstsq(Xc, y_true[cal_ids], rcond=None)[0]
            a, b = float(beta[0]), float(beta[1])

            # apply to ALL external preds, but evaluate ONLY on the test subset
            y_corr    = a + b * y_pred
            mets_pre  = _metrics(y_true[test_ids], y_pred[test_ids])
            mets_post = _metrics(y_true[test_ids], y_corr[test_ids])

            rows_t.append({
                "target": tcol, "repeat": r+1, "a": a, "b": b,
                "rmse_pre": mets_pre["rmse"],  "r2_pre": mets_pre["r2"],
                "rmse_post": mets_post["rmse"], "r2_post": mets_post["r2"],
            })

            # cache details for later plotting
            draw_cache.setdefault(tcol, []).append({
                "repeat": r+1,
                "a": a, "b": b,
                "cal_ids": cal_ids.tolist(),
                "test_ids": test_ids.tolist(),
                "rmse_pre":  mets_pre["rmse"],
                "rmse_post": mets_post["rmse"],
            })

        # ---- write per-target CSVs ----
        df_t = pd.DataFrame(rows_t)
        df_t.to_csv(os.path.join(out_dir, f"{tcol}__calibration_draws.csv"), index=False)

        summary = df_t[["rmse_pre","rmse_post","r2_pre","r2_post"]].agg(["mean","median"])
        summary.to_csv(os.path.join(out_dir, f"{tcol}__calibration_summary.csv"))

        # ---- Visual 1: RMSE histogram (pre vs post) across draws ----
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4), dpi=140)
        bins = max(8, min(30, int(np.sqrt(len(df_t)))))
        plt.hist(df_t["rmse_pre"],  bins=bins, alpha=0.6, label="RMSE pre")
        plt.hist(df_t["rmse_post"], bins=bins, alpha=0.6, label="RMSE post")
        plt.xlabel("RMSE"); plt.ylabel("Count")
        plt.title(f"{nice} — Calibration Transfer (RMSE pre/post)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tcol}__rmse_hist_pre_post.png"), dpi=300)
        plt.close()

        # ---- Visual 2: Parity plots (pre & post) for a representative draw ----
        details = sorted(draw_cache.get(tcol, []), key=lambda d: d["rmse_post"])
        if details:
            rep = details[len(details)//2]  # median by post-RMSE (use details[0] for "best")
            a, b     = rep["a"], rep["b"]
            test_ids = np.array(rep["test_ids"], dtype=int)

            y_true_t = y_true[test_ids]
            y_pred_t = y_pred[test_ids]
            y_sd_t   = y_sd[test_ids]
            y_corr_t = a + b * y_pred_t

            title_suffix = f"Cal transfer (rep draw={rep['repeat']}; a={a:.2f}, b={b:.2f})"
            parity_plot_sample_level(
                y_true_sample=y_true_t.reshape(-1,1),
                y_pred_sample=y_pred_t.reshape(-1,1),
                y_true_sample_std=None,
                y_pred_sample_std=y_sd_t.reshape(-1,1),
                target_names=(nice,),
                out_dir=out_dir,
                fname_prefix=f"{tcol}__parity_pre_repdraw",
                dpi=300,
                title_suffix=title_suffix + " — PRE"
            )
            parity_plot_sample_level(
                y_true_sample=y_true_t.reshape(-1,1),
                y_pred_sample=y_corr_t.reshape(-1,1),
                y_true_sample_std=None,
                y_pred_sample_std=y_sd_t.reshape(-1,1),
                target_names=(nice,),
                out_dir=out_dir,
                fname_prefix=f"{tcol}__parity_post_repdraw",
                dpi=300,
                title_suffix=title_suffix + " — POST"
            )

        print(f"  [{tcol}] wrote draws, summary, and visuals → {out_dir}")

def spxy_repeats_and_mccv():
    print("\n[3/4] Stability scans: SPXY repeats + grouped MCCV")
    wn_tr, avg_tr, all_tr, meta_tr, *_ = load_train_external()
    X_all = all_tr.T
    X_samp = avg_tr.T

    out_dir = os.path.join(OUT_DIR, "stability")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- SPXY repeats ----------
    rows_spxy = []
    for nice, tcol in TARGETS:
        keep = pd.to_numeric(meta_tr[tcol], errors="coerce").notna().to_numpy()
        y_samp = meta_tr.loc[keep, tcol].to_numpy(float)
        Xs = X_samp[keep]
        orig_keep_idx = np.where(keep)[0]

        model, methods, hp, hp_str = winner_for_target(tcol)

        for r in range(SPXY_REPEATS):
            cal_idx, val_idx = _spxy_split(Xs, y_samp, cal_frac=SPXY_CAL_FRAC, alpha=SPXY_ALPHA, seed=RANDOM_STATE+r)
            orig_cal = orig_keep_idx[cal_idx]; orig_val = orig_keep_idx[val_idx]
            row_cal  = (orig_cal[:,None]*REPS + np.arange(REPS)).ravel()
            row_val  = (orig_val[:,None]*REPS + np.arange(REPS)).ravel()

            Xcal = X_all[row_cal]; Xval = X_all[row_val]
            ycal = np.repeat(y_samp[cal_idx].reshape(-1,1), REPS, axis=0)
            yval = y_samp[val_idx]
            groups_cal = _groups_for_spectra(len(cal_idx), reps=REPS)

            yhat_spec = _train_predict(model, methods, hp, Xcal, ycal, Xval, groups_cal)
            yhat, _ = _sample_means_stds(yhat_spec, reps=REPS)
            mets = _metrics(yval, yhat)
            rows_spxy.append({"target": tcol, "repeat": r+1, **mets})

    if rows_spxy:
        df_spxy = pd.DataFrame(rows_spxy)
        df_spxy.to_csv(os.path.join(out_dir, "spxy_repeats_metrics.csv"), index=False)
        print(f"  SPXY repeats written → {out_dir}")

    # ---------- Grouped MCCV ----------
    rows_mccv = []
    per_sample = {}  # target -> dict(sample_index -> list of errors)

    rng = np.random.default_rng(RANDOM_STATE)
    for nice, tcol in TARGETS:
        keep = pd.to_numeric(meta_tr[tcol], errors="coerce").notna().to_numpy()
        y_samp = meta_tr.loc[keep, tcol].to_numpy(float)
        orig = np.where(keep)[0]

        model, methods, hp, hp_str = winner_for_target(tcol)
        per_sample[tcol] = {int(i): [] for i in orig}

        for r in range(MCCV_REPEATS):
            # random validation set at sample level
            n = len(orig)
            n_val = max(2, int(math.ceil(MCCV_VAL_FRAC*n)))
            val_local_idx = np.sort(rng.choice(n, size=n_val, replace=False))
            tr_local_idx  = np.setdiff1d(np.arange(n), val_local_idx)

            orig_val = orig[val_local_idx]; orig_tr = orig[tr_local_idx]

            row_tr  = (orig_tr[:,None]*REPS + np.arange(REPS)).ravel()
            row_val = (orig_val[:,None]*REPS + np.arange(REPS)).ravel()

            Xtr = X_all[row_tr]; Xval = X_all[row_val]
            ytr = np.repeat(y_samp[tr_local_idx].reshape(-1,1), REPS, axis=0)
            yval = y_samp[val_local_idx]

            groups_tr = _groups_for_spectra(len(tr_local_idx), reps=REPS)

            yhat_spec = _train_predict(model, methods, hp, Xtr, ytr, Xval, groups_tr)
            yhat, _ = _sample_means_stds(yhat_spec, reps=REPS)

            # record metrics & per-sample absolute error
            mets = _metrics(yval, yhat)
            rows_mccv.append({"target": tcol, "repeat": r+1, **mets})
            for sid, yt, yp in zip(orig_val, yval, yhat):
                per_sample[tcol][int(sid)].append(float(abs(yt-yp)))

    if rows_mccv:
        df_mccv = pd.DataFrame(rows_mccv)
        df_mccv.to_csv(os.path.join(out_dir, "mccv_metrics.csv"), index=False)

        # sample "hardness" tables
        for nice, tcol in TARGETS:
            recs = []
            for sid, errs in per_sample[tcol].items():
                if errs:
                    recs.append({"sample_index": sid,
                                 "n_seen": len(errs),
                                 "mae_mean": float(np.mean(errs)),
                                 "mae_std":  float(np.std(errs))})
            d = pd.DataFrame(recs).sort_values(["mae_mean","mae_std"], ascending=False)
            d.to_csv(os.path.join(out_dir, f"{tcol}__mccv_hard_samples.csv"), index=False)
        print(f"  MCCV metrics + hard-sample tables written → {out_dir}")

def label_audit():
    print("\n[4/4] Label audit (train vs external)")
    _, _, _, meta_tr, _, _, _, meta_ex = load_train_external()
    out_dir = os.path.join(OUT_DIR, "label_audit")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for nice, tcol in TARGETS:
        yt = pd.to_numeric(meta_tr[tcol], errors="coerce")
        ye = pd.to_numeric(meta_ex[tcol], errors="coerce")

        for name, s in [("train", yt), ("external", ye)]:
            v = s.dropna().astype(float).values
            if v.size == 0: continue
            rows.append({
                "target": tcol, "split": name,
                "n": v.size, "mean": float(np.mean(v)), "std": float(np.std(v)),
                "min": float(np.min(v)), "q25": float(np.percentile(v,25)),
                "median": float(np.median(v)), "q75": float(np.percentile(v,75)),
                "max": float(np.max(v))
            })
            # hist
            plt.figure(figsize=(6,4), dpi=140)
            plt.hist(v, bins=min(10, max(5, int(np.sqrt(v.size)))), alpha=0.85)
            plt.title(f"{nice} — {name} distribution")
            plt.xlabel(nice); plt.ylabel("count"); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{tcol}__{name}_hist.png"), dpi=300); plt.close()

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "label_summary.csv"), index=False)
    print(f"  Wrote label histograms + summary CSV → {out_dir}")

# ---------------------- RUN ALL ----------------------
def main():
    domain_shift_report()
    calibration_transfer_simulation()
    spxy_repeats_and_mccv()
    label_audit()
    print("\n[OK] Diagnostics complete →", OUT_DIR)

if __name__ == "__main__":
    main()
