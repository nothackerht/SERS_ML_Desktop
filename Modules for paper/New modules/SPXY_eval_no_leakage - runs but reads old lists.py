# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 13:20:54 2025

@author: spect
"""

# -*- coding: utf-8 -*-
"""
SPXY evaluation (NO LEAK) with explicit CV-based selection and paper-ready outputs.

What's new vs. your original:
- Winner selection is done ONLY on SPXY-train via inner GroupKFold (no selection on holdout).
- We save both selection metrics (sel_cv_rmse, sel_cv_sd) and validation metrics (val_rmse, val_r2).
- Added 'cv' and 'cv_method' columns to all Excel outputs.
- Output directories now match your requested "Modules for paper" folders for each scenario.

Flip PRESET below to target one of the four scenarios.
"""

import os
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
# <-- put this BEFORE importing your local modules
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from data_loader import load_data
from preprocessing import Preprocessing
from plot_parity import parity_plot_sample_level, save_outer_predictions_excel
from ten_fold_cv_regression_global import _best_interval_grouped

# --- sklearn ---
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    median_absolute_error,
    explained_variance_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold


# --- CLI shim: paths & settings come from main.py ---
import argparse
_cli = argparse.ArgumentParser(add_help=False)
_cli.add_argument("--data_dir")
_cli.add_argument("--meta_path")
_cli.add_argument("--include_types", default="DM1,Control")
_cli.add_argument("--out_dir")
# per-target leaderboard dirs for replaying all combos:
_cli.add_argument("--results_dir_adf")
_cli.add_argument("--results_dir_hgs")
_cli.add_argument("--results_dir_si")
# SPXY seeds + aggregation toggle
_cli.add_argument("--spxy_repeats", type=int, default=10)  # default: 10 SPXY splits
_cli.add_argument("--aggregate_selection_across_seeds", type=int, default=0)  # 1 = enable aggregation
# global random state
_cli.add_argument("--random_state", type=int, default=42)

args, _ = _cli.parse_known_args()

TRAIN_DATA_DIR   = args.data_dir
TRAIN_META_PATH  = args.meta_path
INCLUDE_TYPES    = tuple([x.strip() for x in (args.include_types or "DM1,Control").split(",") if x.strip()])
OUT_DIR_SPXY     = args.out_dir
os.makedirs(OUT_DIR_SPXY, exist_ok=True)

BASE_SEED = int(getattr(args, "random_state", 42))

# build from raw args first
_raw_results_dirs = {
    "ADF_pp_avg": args.results_dir_adf,
    "HGS_pp_avg": args.results_dir_hgs,
    "target_SI":  args.results_dir_si,
}

# warn early, then normalize to Path only when present
for k, v in _raw_results_dirs.items():
    if not v:
        print(f"[WARN] results_dir for {k} not provided.")
    elif not Path(v).exists():
        print(f"[WARN] results_dir for {k} does not exist: {v}")

RESULT_DIRS_MAP = {k: Path(v) for k, v in _raw_results_dirs.items() if v}

# seeds / repeats
N_REPEATS           = int(args.spxy_repeats or 1)
AGGREGATE_SELECTION = bool(int(args.aggregate_selection_across_seeds or 0))






TARGETS = [
    ("Splicing Index", "target_SI"),
    ("Hand Grip Strength (%)", "HGS_pp_avg"),
    ("Average Ankle Dorsiflexion (%)", "ADF_pp_avg"),
]

REPS = 9             # spectra per sample
CAL_FRAC = 0.80      # SPXY calibration fraction
ALPHA = 0.5          # SPXY weighting between X and y distances

INNER_FOLDS = 5      # inner CV folds on SPXY-train

# ====================== HELPERS ======================

def _groups_for_spectra(n_samples: int, reps: int = 9) -> np.ndarray:
    import numpy as _np
    return _np.repeat(_np.arange(n_samples, dtype=int), reps)

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
    if label is None:
        return []
    s = str(label).strip()
    if not s or s.lower() in ("nan", "none") or s.lower().startswith("no"):
        return []
    return [p.strip() for p in s.split("+")]

def _parse_hp(model: str, hp_str: str):
    model = model.lower().strip()
    if model == "pls":
        m = re.match(r"C=(\d+)", hp_str.strip());  assert m, f"Bad PLS hp: {hp_str}"
        return {"n_components": int(m.group(1))}
    if model == "ipls":
        m = re.match(r"C=(\d+)__I=(\d+)", hp_str.strip());  assert m, f"Bad iPLS hp: {hp_str}"
        return {"n_components": int(m.group(1)), "num_intervals": int(m.group(2))}
    if model == "rf":
        patt = r"ne(\d+)_md(None|\d+)_mss(\d+)_msl(\d+)_mf(\w+)"
        m = re.match(patt, hp_str.strip());        assert m, f"Bad RF hp: {hp_str}"
        ne, md_raw, mss, msl, mf = m.groups()
        md_val = None if md_raw.lower() == "none" else int(md_raw)
        return {"n_estimators": int(ne), "max_depth": md_val,
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

def _train_predict(model_name, methods, hp, Xcal, ycal, Xval, groups_cal):
    """
    Train on Xcal/ycal and predict on Xval.

    Returns
    -------
    y_pred_spec : np.ndarray
        Spectrum-level predictions for Xval.
    interval_info : dict or None
        For iPLS models: {"a": int, "b": int} giving the selected column indices.
        For other models: None.
    """
    model_name = model_name.lower().strip()
    interval_info = None

    if model_name == "pls":
        Xt_tr, Xt_va = _fit_transform_pair(Xcal, Xval, methods)
        mdl = PLSRegression(n_components=int(hp["n_components"]))
        mdl.fit(Xt_tr, ycal)
        return mdl.predict(Xt_va).ravel(), interval_info

    if model_name == "rf":
        Xt_tr, Xt_va = _fit_transform_pair(Xcal, Xval, methods)
        rf = RandomForestRegressor(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            min_samples_split=hp["min_samples_split"], min_samples_leaf=hp["min_samples_leaf"],
            max_features=hp["max_features"], random_state=BASE_SEED, n_jobs=-1
        )
        rf.fit(Xt_tr, ycal.ravel())
        return rf.predict(Xt_va), interval_info

    if model_name == "acfnn":
        Xt_tr, Xt_va = _fit_transform_pair(Xcal, Xval, methods)
        mlp = MLPRegressor(random_state=BASE_SEED, **hp)
        mlp.fit(Xt_tr, ycal.ravel())
        return mlp.predict(Xt_va), interval_info

    if model_name == "ipls":
        num_intervals = int(hp["num_intervals"]); n_components = int(hp["n_components"])
        (a, b), _ = _best_interval_grouped(
            X_tr=Xcal, Y_tr=ycal, groups_tr=groups_cal,
            preprocess_methods=methods, n_components=n_components, num_intervals=num_intervals,
            reps=REPS, n_splits=max(3, min(5, int(len(np.unique(groups_cal))))),
            random_state=BASE_SEED
        )
        interval_info = {"a": int(a), "b": int(b)}
        Xt_tr, Xt_va = _fit_transform_pair(Xcal[:, a:b], Xval[:, a:b], methods)
        mdl = PLSRegression(n_components=n_components)
        mdl.fit(Xt_tr, ycal)
        return mdl.predict(Xt_va).ravel(), interval_info

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
    """
    SPXY at the SAMPLE level using averaged spectra per sample.
    Returns (cal_idx, val_idx) arrays of sample indices.
    """
    rng = np.random.default_rng(seed)
    N = Xsamp.shape[0]
    Ncal = max(2, int(np.floor(cal_frac * N)))

    # Scale features for distance fairness
    Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(Xsamp)
    yz = StandardScaler(with_mean=True, with_std=True).fit_transform(ysamp.reshape(-1,1)).ravel()

    # Distance matrices
    DX = np.sqrt(((Xz[:,None,:] - Xz[None,:,:])**2).sum(axis=2))
    DY = np.abs(yz[:,None] - yz[None,:])
    # Normalize to [0,1]
    DX = DX / (DX.max() + 1e-12)
    DY = DY / (DY.max() + 1e-12)
    D  = alpha*DX + (1-alpha)*DY

    # Start with the farthest pair; break ties randomly
    i0, j0 = np.unravel_index(np.argmax(D), D.shape)
    if i0 == j0:
        i0 = rng.integers(0, N)
        j0 = (i0 + rng.integers(1, N)) % N

    selected = [i0, j0]
    remaining = set(range(N)) - set(selected)

    while len(selected) < Ncal and remaining:
        r_star = max(remaining, key=lambda r: np.min(D[r, selected]))
        selected.append(r_star)
        remaining.remove(r_star)

    cal_idx = np.array(sorted(selected), dtype=int)
    val_idx = np.array(sorted(list(remaining)), dtype=int)
    assert len(cal_idx) + len(val_idx) == N
    return cal_idx, val_idx

def _load_combos_from_csv_dir(results_dir: Path) -> pd.DataFrame:
    """
    Load distinct (model, preprocessing, hp) rows from leaderboard files only:
    * *_metrics*all_combos*.csv/.xlsx/.xls
    Falls back to any file with the required columns if no leaderboard found.
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    print(f"[INFO] Scanning for leaderboard files in: {results_dir}")

    # Prefer explicit leaderboard filenames to avoid noisy per-HP files.
    preferred = (
        list(results_dir.rglob("*metrics*all_combos*.csv")) +
        list(results_dir.rglob("*metrics*all_combos*.xlsx")) +
        list(results_dir.rglob("*metrics*all_combos*.xls"))
    )
    candidates = preferred or (
        list(results_dir.rglob("*.csv")) +
        list(results_dir.rglob("*.xlsx")) +
        list(results_dir.rglob("*.xls"))
    )
    if not candidates:
        raise FileNotFoundError(f"No CSV/XLSX files found under {results_dir}")

    # Column alias sets
    model_aliases = {"model", "Model", "model_name", "ModelName"}
    prep_aliases  = {"preprocessing", "Preprocessing", "prep", "preproc", "Preproc"}
    hp_aliases    = {"hp", "HP", "hp_tag", "params", "hyperparams", "hyper_params", "HP_Tag"}

    def pick_col(df, aliases):
        cols_lower = {c.lower(): c for c in df.columns}
        for a in aliases:
            if a in df.columns:
                return a
            if a.lower() in cols_lower:
                return cols_lower[a.lower()]
        return None

    collected, scanned, kept = [], 0, 0
    for p in candidates:
        scanned += 1
        try:
            df = pd.read_csv(p) if p.suffix.lower()==".csv" else pd.read_excel(p)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
            continue

        c_model = pick_col(df, model_aliases)
        c_prep  = pick_col(df, prep_aliases)
        c_hp    = pick_col(df, hp_aliases)

        if not (c_model and c_prep and c_hp):
            # Only print SKIP for non-preferred scans to reduce noise
            if not preferred:
                print(f"[SKIP] {p.name} missing required columns (model:{bool(c_model)} prep:{bool(c_prep)} hp:{bool(c_hp)})")
            continue

        sub = df[[c_model, c_prep, c_hp]].copy()
        for c in (c_model, c_prep, c_hp):
            sub[c] = sub[c].astype(str).str.strip()

        collected.append(sub.rename(columns={c_model:"model", c_prep:"preprocessing", c_hp:"hp"}))
        kept += len(sub)

        # If we found a leaderboard file, no need to read more
        if preferred:
            break

    if not collected:
        raise FileNotFoundError(
            f"No suitable files with (model, preprocessing, hp) under {results_dir}. "
            f"Scanned {scanned} files; kept 0."
        )

    combos = (pd.concat(collected, ignore_index=True)
              .dropna(subset=["model","preprocessing","hp"])
              .drop_duplicates()
              .reset_index(drop=True))
    print(f"[INFO] Loaded {len(combos)} unique combos from {results_dir} "
          f"(kept {kept} rows from {len(collected)} file(s); scanned {scanned} files total)")
    return combos

# ====================== NEW: inner-CV scoring & tie-breakers ======================

def _inner_cv_score(model_name, methods, hp, Xcal, ycal, groups_cal, n_splits=INNER_FOLDS, seed=None):
    """
    Return mean±sd RMSE from GroupKFold on SPXY-train only.
    Splits respect sample grouping via groups_cal.
    Preprocessing and (for iPLS) interval selection are learned on fold-train only.
    """
    if seed is None:
        seed = BASE_SEED

    gkf = GroupKFold(n_splits=n_splits)
    rmses = []

    for tr_idx, va_idx in gkf.split(Xcal, ycal, groups=groups_cal):
        Xtr, Xva = Xcal[tr_idx], Xcal[va_idx]
        ytr, yva = ycal[tr_idx].ravel(), ycal[va_idx].ravel()

        # fold-specific groups for iPLS interval selection
        fold_groups_tr = groups_cal[tr_idx]

        if model_name.lower() == "ipls":
            (a,b), _ = _best_interval_grouped(
                X_tr=Xtr, Y_tr=ytr.reshape(-1,1), groups_tr=fold_groups_tr,
                preprocess_methods=methods,
                n_components=int(hp["n_components"]),
                num_intervals=int(hp["num_intervals"]),
                reps=REPS,
                n_splits=max(3, min(5, int(len(np.unique(fold_groups_tr))))),
                random_state=seed
            )
            Xt_tr, Xt_va = _fit_transform_pair(Xtr[:, a:b], Xva[:, a:b], methods)
            mdl = PLSRegression(n_components=int(hp["n_components"]))
            mdl.fit(Xt_tr, ytr)
            yva_pred_spec = mdl.predict(Xt_va).ravel()
        else:
            Xt_tr, Xt_va = _fit_transform_pair(Xtr, Xva, methods)
            if model_name.lower() == "pls":
                mdl = PLSRegression(n_components=int(hp["n_components"]))
            elif model_name.lower() == "rf":
                mdl = RandomForestRegressor(
                    n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                    min_samples_split=hp["min_samples_split"], min_samples_leaf=hp["min_samples_leaf"],
                    max_features=hp["max_features"], random_state=seed, n_jobs=-1
                )
            elif model_name.lower() == "acfnn":
                mdl = MLPRegressor(random_state=seed, **hp)
            else:
                raise ValueError(f"Unsupported model in CV: {model_name}")
            mdl.fit(Xt_tr, ytr)
            yva_pred_spec = mdl.predict(Xt_va).ravel()

        # collapse spectra → sample for fold-val
        yva_pred, _ = _sample_means_stds(yva_pred_spec, reps=REPS)
        yva_true, _ = _sample_means_stds(yva, reps=REPS)

        rmses.append(np.sqrt(mean_squared_error(yva_true, yva_pred)))

    return float(np.mean(rmses)), float(np.std(rmses))


def _is_simpler(model_a, hp_a, model_b, hp_b):
    """Simplicity ordering to break ties within ~1-SE band."""
    mA, mB = model_a.lower(), model_b.lower()
    if mA != mB:
        order = {"pls": 0, "ipls": 1, "rf": 2, "acfnn": 3}
        return order.get(mA, 99) < order.get(mB, 99)
    if mA == "pls":
        return int(hp_a["n_components"]) < int(hp_b["n_components"])
    if mA == "ipls":
        ca, ia = int(hp_a["n_components"]), int(hp_a["num_intervals"])
        cb, ib = int(hp_b["n_components"]), int(hp_b["num_intervals"])
        return (ia, ca) < (ib, cb)
    if mA == "rf":
        return int(hp_a["n_estimators"]) < int(hp_b["n_estimators"])
    if mA == "acfnn":
        ha = hp_a.get("hidden_layer_sizes", (128, 64))
        hb = hp_b.get("hidden_layer_sizes", (128, 64))
        return (sum(ha), len(ha)) < (sum(hb), len(hb))
    return False

# ====================== MAIN ======================

def main():
    # Load data
    wn, avg_tr, all_tr, meta = load_data(
        data_dir=TRAIN_DATA_DIR,
        metadata_path=TRAIN_META_PATH,
        include_types=INCLUDE_TYPES,   # DM1 vs DM1+Control
        return_filenames=False,
        strict=False,
        report_samples=3,
    )

    # scikit rows = samples
    X_all = all_tr.T               # spectra-level (N_specs x features)
    X_sample = avg_tr.T            # sample-averaged spectra for SPXY distances

    summary_rows = []

    for nice, tcol in TARGETS:
        print(f"\n=== SPXY (no-leak) for {nice} ({tcol}) ===")

        # per-target containers
        spxy_split_rows = []         # CAL / VAL composition across repeats
        all_spxy_metrics_rows = []   # metrics tables across repeats

        # per-target output root
        tgt_dir = os.path.join(OUT_DIR_SPXY, tcol)
        os.makedirs(tgt_dir, exist_ok=True)
        pred_root = os.path.join(tgt_dir, "all_combos_spxy")
        os.makedirs(pred_root, exist_ok=True)

        # mask samples with valid labels for this target
        keep = pd.to_numeric(meta[tcol], errors="coerce").notna().to_numpy()
        y_sample = meta.loc[keep, tcol].to_numpy(float)
        Xs = X_sample[keep]
        orig_keep_idx = np.where(keep)[0]

        # combos source for this target (explicit per your mapping)
        results_dir = RESULT_DIRS_MAP.get(tcol)
        if results_dir is None:
            raise FileNotFoundError(f"No results_dir provided for {tcol}.")

        combos_df = _load_combos_from_csv_dir(results_dir)

        # Filter to supported model types only
        allowed_models = {"pls", "rf", "acfnn", "ipls"}
        combos_df["model"] = combos_df["model"].astype(str).str.strip()
        invalid = sorted(set(combos_df["model"].str.lower()) - allowed_models)
        if invalid:
            print(f"[WARN] Dropping combos with unsupported models for {tcol}: {invalid}")
        combos_df = combos_df[combos_df["model"].str.lower().isin(allowed_models)].reset_index(drop=True)
        if combos_df.empty:
            raise ValueError(f"No valid (model, preprocessing, hp) combos found for {tcol} after filtering.")

        # ---------- SPXY repeats ----------
        for r in range(N_REPEATS):
            # SPXY split at the SAMPLE level
            cal_idx_s, val_idx_s = _spxy_split(
                Xs, y_sample,
                cal_frac=CAL_FRAC,
                alpha=ALPHA,
                seed=BASE_SEED + r,
            )

            # Debug: report SPXY split sizes
            print(f"[SPXY] {tcol} | repeat {r+1}: "
                  f"N_valid={len(y_sample)}, Ncal={len(cal_idx_s)}, Nval={len(val_idx_s)}")

            # Log which samples are in CAL / VAL for this target & repeat
            for role, idx_array in (("CAL", cal_idx_s), ("VAL", val_idx_s)):
                for sidx in idx_array:
                    gidx = int(orig_keep_idx[sidx])
                    row_meta = meta.iloc[gidx]
                    spxy_split_rows.append({
                        "target": tcol,
                        "repeat": r+1,
                        "role": role,                  # CAL or VAL
                        "sample_idx_local": int(sidx), # index within 'keep' subset
                        "sample_idx_global": gidx,     # index within full meta
                        "SampleID": row_meta.get("SampleID", None),
                        "Type": row_meta.get("Type", None),
                    })

            # ---------- expand SPXY split to spectra ----------
            orig_cal_idx = orig_keep_idx[cal_idx_s]
            orig_val_idx = orig_keep_idx[val_idx_s]

            row_idx_cal = (orig_cal_idx[:, None] * REPS + np.arange(REPS)).ravel()
            row_idx_val = (orig_val_idx[:, None] * REPS + np.arange(REPS)).ravel()

            Xcal = X_all[row_idx_cal]
            Xval = X_all[row_idx_val]

            ycal = np.repeat(y_sample[cal_idx_s].reshape(-1, 1), REPS, axis=0)
            yval_true = y_sample[val_idx_s]
            groups_cal = _groups_for_spectra(len(cal_idx_s), reps=REPS)

            # ---------- evaluate all combos for this repeat ----------
            spxy_rows = []
            best = None  # will hold a dict of the current best-by-selection

            for _, row in combos_df.iterrows():
                model_name = str(row["model"]).strip().lower()
                prep_label = str(row["preprocessing"])
                hp_str     = str(row["hp"])

                methods = _parse_methods(prep_label)
                hp      = _parse_hp(model_name, hp_str)

                # 1) selection on training only (inner GroupKFold)
                cv_rmse, cv_sd = _inner_cv_score(
                    model_name, methods, hp,
                    Xcal, ycal, groups_cal,
                    n_splits=INNER_FOLDS,
                    seed=BASE_SEED,
                )

                # 2) lock combo → evaluate once on SPXY holdout
                yval_hat_spec, interval_info = _train_predict(
                    model_name, methods, hp, Xcal, ycal, Xval, groups_cal
                )
                yval_pred, yval_pred_sd = _sample_means_stds(yval_hat_spec, reps=REPS)
                mets = _metrics(yval_true, yval_pred)


                # save per-combo predictions under a model/prep folder
                methods_key = '+'.join([m.replace(' ', '_') for m in methods]) if methods else 'none'
                model_dir = os.path.join(pred_root, f"{model_name}__{methods_key}")
                os.makedirs(model_dir, exist_ok=True)
                out_xlsx = os.path.join(model_dir, f"{hp_str}.xlsx")
                pd.DataFrame({
                    f"{tcol}__true": yval_true,
                    f"{tcol}__pred": yval_pred
                }).to_excel(out_xlsx, index=False)

                # Default interval fields (for non-iPLS models)
                interval_start_idx = np.nan
                interval_end_idx   = np.nan
                interval_start_wn  = np.nan
                interval_end_wn    = np.nan
            
                # If this is an iPLS model, fill in the chosen interval and wavenumbers
                if model_name == "ipls" and interval_info is not None:
                    a = int(interval_info["a"])
                    b = int(interval_info["b"])
                    interval_start_idx = a
                    interval_end_idx   = b
                    interval_start_wn  = float(wn[a])
                    interval_end_wn    = float(wn[b - 1])
            
                spxy_rows.append({
                    "repeat": r+1,
                    "target": tcol,
                    "model": model_name,
                    "preprocessing": prep_label,
                    "hp": hp_str,
                    "cv": True,
                    "cv_method": f"GroupKFold({INNER_FOLDS}) on SPXY-train",
                    "sel_cv_rmse": cv_rmse,
                    "sel_cv_sd": cv_sd,
                    "val_rmse": mets["rmse"],
                    "val_r2": mets["r2"],
                    "val_mae": mets["mae"],
                    "val_medae": mets["medae"],
                    "val_evs": mets["evs"],
                    "interval_start_idx": interval_start_idx,
                    "interval_end_idx": interval_end_idx,
                    "interval_start_wn": interval_start_wn,
                    "interval_end_wn": interval_end_wn,
                })

                # 3) track winner BY TRAIN-ONLY EVIDENCE (1-SE toward simplicity)
                if best is None:
                    best = {
                        "sel_cv_rmse": cv_rmse, "sel_cv_sd": cv_sd,
                        "model": model_name, "methods": methods, "hp": hp_str,
                        "y_pred_best": yval_pred, "y_pred_sd_best": yval_pred_sd,
                        "val_rmse": mets["rmse"], "val_r2": mets["r2"],
                        "interval_info": interval_info if model_name == "ipls" else None,
                    }

                else:
                    thresh = best["sel_cv_rmse"] + best["sel_cv_sd"]
                    is_simpler = _is_simpler(
                        model_name, hp,
                        best["model"], _parse_hp(best["model"], best["hp"])
                    )
                    if (cv_rmse < best["sel_cv_rmse"] - 1e-12) or \
                       (cv_rmse <= thresh and is_simpler):
                        best.update({
                            "sel_cv_rmse": cv_rmse, "sel_cv_sd": cv_sd,
                            "model": model_name, "methods": methods, "hp": hp_str,
                            "y_pred_best": yval_pred, "y_pred_sd_best": yval_pred_sd,
                            "val_rmse": mets["rmse"], "val_r2": mets["r2"],
                            "interval_info": interval_info if model_name == "ipls" else None,
                        })


            # write consolidated SPXY metrics table for this target & repeat
            if spxy_rows:
                spxy_metrics = (
                    pd.DataFrame(spxy_rows)
                    .sort_values(by=["sel_cv_rmse", "model", "preprocessing", "hp"])
                    .reset_index(drop=True)
                )
            else:
                spxy_metrics = pd.DataFrame(columns=[
                    "repeat","target","model","preprocessing","hp","cv","cv_method",
                    "sel_cv_rmse","sel_cv_sd","val_rmse","val_r2","val_mae","val_medae","val_evs"
                ])

            spxy_metrics_path = os.path.join(
                tgt_dir, f"{tcol}__metrics_all_combos_SPXY_NO_LEAK_r{r+1}.xlsx"
            )
            spxy_metrics.to_excel(spxy_metrics_path, index=False)
            print(f"[SPXY-METRICS] wrote {spxy_metrics_path}  ({len(spxy_metrics)} combos)")
            all_spxy_metrics_rows.append(spxy_metrics.copy())

            # winner artifacts for this repeat
            if best is not None:
                model_best      = best["model"]
                methods_best    = best["methods"]
                hp_best         = best["hp"]
                y_pred_best     = best["y_pred_best"]
                y_pred_sd_best  = best["y_pred_sd_best"]

                methods_label = " + ".join(methods_best) if methods_best else "No Preprocessing"
                methods_path  = "+".join(m.replace(" ","_") for m in methods_best) if methods_best else "none"
                title_suffix  = (
                    f"{model_best.upper()} | {methods_label} | HP: {hp_best} | "
                    f"SPXY {int(CAL_FRAC*100)}/{int((1-CAL_FRAC)*100)} | "
                    f"Selected by inner-CV (RMSE={best['sel_cv_rmse']:.3f}±{best['sel_cv_sd']:.3f})"
                )

                winner_dir = os.path.join(tgt_dir, f"winner_SPXY__{model_best}__{methods_path}")
                os.makedirs(winner_dir, exist_ok=True)

                parity_plot_sample_level(
                    y_true_sample=yval_true.reshape(-1,1),
                    y_pred_sample=np.asarray(y_pred_best).reshape(-1,1),
                    y_true_sample_std=None,
                    y_pred_sample_std=np.asarray(y_pred_sd_best).reshape(-1,1),
                    target_names=(nice,),
                    out_dir=winner_dir,
                    fname_prefix=f"parity_spxy_r{r+1}",
                    title_suffix=title_suffix,
                )

                # tidy predictions for the winner (with metadata in VAL order)
                meta_kept = meta.loc[keep].reset_index(drop=True)
                meta_val = meta_kept.iloc[val_idx_s].reset_index(drop=True)
                save_outer_predictions_excel(
                    y_true_sample=yval_true.reshape(-1,1),
                    y_pred_sample=np.asarray(y_pred_best).reshape(-1,1),
                    meta_kept=meta_val,
                    target_order=(tcol,),
                    out_path=os.path.join(
                        winner_dir,
                        f"{tcol}__winner_SPXY_predictions_NO_LEAK_r{r+1}.xlsx"
                    ),
                    y_true_sample_std=None,
                    y_pred_sample_std=np.asarray(y_pred_sd_best).reshape(-1,1),
                )

                # add run summary (best-of-repeat)
                interval_start_idx = np.nan
                interval_end_idx   = np.nan
                interval_start_wn  = np.nan
                interval_end_wn    = np.nan
                
                if best["model"] == "ipls" and best.get("interval_info") is not None:
                    a = int(best["interval_info"]["a"])
                    b = int(best["interval_info"]["b"])
                    interval_start_idx = a
                    interval_end_idx   = b
                    interval_start_wn  = float(wn[a])
                    interval_end_wn    = float(wn[b - 1])
                
                summary_rows.append({
                    "target": tcol,
                    "repeat": r+1,
                    "cv": True,
                    "cv_method": f"GroupKFold({INNER_FOLDS}) on SPXY-train",
                    "best_model": best["model"],
                    "best_preprocessing": ("+".join(best["methods"]) if best["methods"] else "None"),
                    "best_hp": hp_best,
                    "sel_cv_rmse": float(best["sel_cv_rmse"]),
                    "sel_cv_sd": float(best["sel_cv_sd"]),
                    "val_rmse": float(best["val_rmse"]),
                    "val_r2": float(best["val_r2"]),
                    "interval_start_idx": interval_start_idx,
                    "interval_end_idx": interval_end_idx,
                    "interval_start_wn": interval_start_wn,
                    "interval_end_wn": interval_end_wn,
                })


        # ---------- after all repeats for this target ----------

        # Save SPXY split documentation
        if spxy_split_rows:
            df_splits = pd.DataFrame(spxy_split_rows)
            splits_path = os.path.join(tgt_dir, f"{tcol}__SPXY_splits_NO_LEAK.xlsx")
            df_splits.to_excel(splits_path, index=False)
            print(f"[SPXY-SPLITS] wrote {splits_path} ({len(df_splits)} rows)")

        # Aggregate selection metrics across SPXY repeats (if requested)
        if AGGREGATE_SELECTION and all_spxy_metrics_rows:
            df_all_rep = pd.concat(all_spxy_metrics_rows, ignore_index=True)
            grouped = df_all_rep.groupby(
                ["target", "model", "preprocessing", "hp"], as_index=False
            ).agg({
                "sel_cv_rmse": ["mean", "std"],
                "sel_cv_sd":   ["mean"],
                "val_rmse":    ["mean", "std"],
                "val_r2":      ["mean", "std"],
                "val_mae":     ["mean"],
                "val_medae":   ["mean"],
                "val_evs":     ["mean"],
            })

            # flatten multiindex columns
            grouped.columns = [
                "_".join([c for c in col if c]) if isinstance(col, tuple) else col
                for col in grouped.columns.to_list()
            ]

            agg_path = os.path.join(tgt_dir, f"{tcol}__SPXY_aggregate_across_repeats_NO_LEAK.xlsx")
            grouped.to_excel(agg_path, index=False)
            print(f"[SPXY-AGG] wrote aggregated metrics across repeats → {agg_path}")

    # write summary across targets / repeats
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        out_sum = os.path.join(OUT_DIR_SPXY, "spxy_summary_NO_LEAK.xlsx")
        df_sum.to_excel(out_sum, index=False)
        print(f"\n[OK] Wrote SPXY summary → {out_sum}")


if __name__ == "__main__":
    main()
