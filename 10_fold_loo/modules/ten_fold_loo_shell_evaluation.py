# -*- coding: utf-8 -*-
"""
10_fold_loo_shell_evaluation.py

Leave-One-Out evaluation on a blinded test set using 10-fold GroupKFold,
with 5-fold inner CV for hyperparameter tuning (only hyperparams, fixed preprocessing).
Runs each model over each preprocessing chain separately, avoiding data leakage.
Saves per-model, per-prep, per-target .csv and parity plots (with error bars, hyperparams & prep).
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

from modules.interval_shell import get_model_by_name
from modules.data_loader import load_data, load_metadata
from modules.preprocessing import Preprocessing

def leave_one_out_test_evaluation(
    train_data_dir: str,
    train_meta_path: str,
    test_data_dir: str,
    test_meta_path: str,
    model_list: list[str],
    hyperparam_grids: dict[str, list[dict]],
    preprocess_grid: list[list[str]],
    output_dir: str,
    target_column: str = "target_SI"
):
    os.makedirs(output_dir, exist_ok=True)

    # Load training and test sets
    _, _, all_train = load_data(train_data_dir, metadata_path=train_meta_path)
    meta_train     = load_metadata(train_meta_path)
    _, _, all_test  = load_data(test_data_dir)
    meta_test      = load_metadata(test_meta_path)

    n_train     = all_train.shape[1] // 9
    n_test      = all_test.shape[1] // 9
    test_groups = np.repeat(np.arange(n_test), 9)

    results = []

    # Outer LOO per sample
    for model_name in model_list:
        for prep_chain in preprocess_grid:
            outer = GroupKFold(n_splits=n_test)
            for fold, (ti, to) in enumerate(outer.split(all_test.T, groups=test_groups), 1):
                held = np.unique(test_groups[to])[0]

                # Build combined raw spectra and labels
                X_raw = np.hstack([all_train, all_test[:, ti]]).T       # (n_spectra_total, n_wavenumbers)
                y_vals = np.hstack([
                    np.repeat(meta_train[target_column].values, 9),
                    np.repeat(meta_test[target_column].values[np.setdiff1d(np.arange(n_test), held)], 9)
                ])                                                        # (n_spectra_total,)
                groups = np.repeat(np.arange(n_train + n_test - 1), 9)

                # Inner CV for hyperparameter tuning
                best_rmse = np.inf
                best_hp   = None
                inner     = GroupKFold(n_splits=5)
                for hp in hyperparam_grids.get(model_name, []):
                    fold_rmses = []
                    for iti, ito in inner.split(X_raw, y_vals, groups=groups):
                        # split raw spectra
                        X_tr_raw = X_raw[iti].T
                        X_vl_raw = X_raw[ito].T
                        y_tr      = y_vals[iti]
                        y_vl      = y_vals[ito]

                        # apply preprocessing chain
                        Xp_tr = X_tr_raw.copy()
                        Xp_vl = X_vl_raw.copy()
                        for method in prep_chain:
                            if method == 'EMSC':
                                ref   = np.mean(Xp_tr, axis=1)
                                Xp_tr = Preprocessing(None).emsc(Xp_tr, reference=ref)
                                Xp_vl = Preprocessing(None).emsc(Xp_vl, reference=ref)
                            elif method == 'SNV':
                                Xp_tr = Preprocessing(Xp_tr).snv(Xp_tr)
                                Xp_vl = Preprocessing(Xp_vl).snv(Xp_vl)
                            elif method == 'Normalization':
                                Xp_tr = Preprocessing(Xp_tr).normalize_spectrum(Xp_tr)
                                Xp_vl = Preprocessing(Xp_vl).normalize_spectrum(Xp_vl)
                            elif method == 'Second Derivative':
                                Xp_tr = Preprocessing(Xp_tr).second_derivative(Xp_tr)
                                Xp_vl = Preprocessing(Xp_vl).second_derivative(Xp_vl)

                        Xp_tr = Xp_tr.T
                        Xp_vl = Xp_vl.T

                        # fit & predict on val
                        mdl = get_model_by_name(model_name, hp)
                        mdl.fit(Xp_tr, y_tr.reshape(-1,1))
                        preds = mdl.predict(Xp_vl).flatten()

                        # sample-level RMSE
                        s_true = y_vl.reshape(-1,9).mean(axis=1)
                        s_pred = preds.reshape(-1,9).mean(axis=1)
                        fold_rmses.append(np.sqrt(mean_squared_error(s_true, s_pred)))

                    avg_rmse = np.mean(fold_rmses)
                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_hp   = hp

                # retrain on full combined + predict held-out nine spectra
                # preprocess full combined
                X_full_raw = X_raw.T
                Xp_full    = X_full_raw.copy()
                for method in prep_chain:
                    if method == 'EMSC':
                        ref     = np.mean(Xp_full, axis=1)
                        Xp_full = Preprocessing(None).emsc(Xp_full, reference=ref)
                    elif method == 'SNV':
                        Xp_full = Preprocessing(Xp_full).snv(Xp_full)
                    elif method == 'Normalization':
                        Xp_full = Preprocessing(Xp_full).normalize_spectrum(Xp_full)
                    elif method == 'Second Derivative':
                        Xp_full = Preprocessing(Xp_full).second_derivative(Xp_full)
                Xp_full = Xp_full.T

                # preprocess held-out block (9 spectra)
                loo_raw = all_test[:, to]  # (n_wavenumbers,9)
                Xp_loo = loo_raw.copy()
                for method in prep_chain:
                    if method == 'EMSC':
                        ref    = np.mean(X_full_raw, axis=1)
                        Xp_loo = Preprocessing(None).emsc(Xp_loo, reference=ref)
                    elif method == 'SNV':
                        Xp_loo = Preprocessing(Xp_loo).snv(Xp_loo)
                    elif method == 'Normalization':
                        Xp_loo = Preprocessing(Xp_loo).normalize_spectrum(Xp_loo)
                    elif method == 'Second Derivative':
                        Xp_loo = Preprocessing(Xp_loo).second_derivative(Xp_loo)
                Xp_loo = Xp_loo.T  # (9, n_wavenumbers)

                mdl = get_model_by_name(model_name, best_hp)
                mdl.fit(Xp_full, y_vals.reshape(-1,1))
                raw_preds = mdl.predict(Xp_loo).flatten()  # list of 9

                # store all 9 preds plus mean/std
                pred_mean = raw_preds.mean()
                pred_std  = raw_preds.std()

                results.append({
                    'fold':       fold,
                    'model':      model_name,
                    'preprocess': '+'.join(prep_chain),
                    'hyperparams': best_hp,
                    'held_out':   held,
                    'predictions': raw_preds.tolist(),
                    'pred_mean':  pred_mean,
                    'pred_std':   pred_std,
                    'true':       meta_test[target_column].values[held]
                })

    # assemble dataframe
    df = pd.DataFrame(results)
    df['hyperparams'] = df['hyperparams'].apply(json.dumps)
    df.to_csv(os.path.join(output_dir, 'loo_predictions.csv'), index=False)

    # sample-level metrics
    metrics = df.groupby(['model','preprocess']).apply(
        lambda g: pd.Series({
            'RMSE': np.sqrt(mean_squared_error(g['true'], g['pred_mean'])),
            'R2':   r2_score(g['true'], g['pred_mean'])
        })
    )
    metrics.to_csv(os.path.join(output_dir, 'loo_metrics.csv'))

    # parity plots
    for (m, prep), grp in df.groupby(['model','preprocess']):
        plt.figure(figsize=(6,6))
        plt.errorbar(
            grp['true'],
            grp['pred_mean'],
            yerr=grp['pred_std'], fmt='o', capsize=4
        )
        mn, mx = grp['true'].min(), grp['true'].max()
        plt.plot([mn,mx],[mn,mx],'r--')
        plt.xlabel(target_column, fontweight='bold')
        plt.ylabel(target_column, fontweight='bold')
        plt.title(f"{m} | Preproc: {prep}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"parity_{m}_{prep.replace('+','_')}.png"), dpi=300)
        plt.show()
        plt.close()

    return df, metrics

if __name__ == "__main__":
    # ────────────────────────────────────────────────────────────────
    # 10-Fold LOO (example invocation with your new paths)
    # ────────────────────────────────────────────────────────────────
    from modules.ten_fold_loo_shell_evaluation import leave_one_out_test_evaluation

    # 1) Where to read train & test
    train_data_dir     = r"C:\Users\notha\Downloads\Data\data"
    train_meta_path    = r"C:\Users\notha\Downloads\Data\y_metadata.csv"
    test_data_dir      = r"C:\Users\notha\Downloads\Data\data_test_updated"
    test_meta_path     = r"C:\Users\notha\Downloads\Data\y_metadata_test_updated.csv"

    # 2) Where to write results
    results_root       = r"C:\Users\notha\Downloads\Data\Results"

    # 3) Which models / hyper-grids / preprocess chains to try
    model_list         = ['sipls']
    hyperparam_grids   = {
        'sipls': [
            {'n_components': 2, 'device': 'cpu'},
            # you can add more here…
        ]
    }
    preprocess_grid    = [[]]  # just “no preprocessing” for now

    # 4) Targets
    targets = [("Splicing Index", "target_SI")]

    # 5) Launch
    for nice_name, col in targets:
        out_dir = os.path.join(results_root, f"10_fold_LOO_{col}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n\n### Running LOO for {nice_name} ({col}) → {out_dir}")
        loo_df, loo_metrics = leave_one_out_test_evaluation(
            train_data_dir   = train_data_dir,
            train_meta_path  = train_meta_path,
            test_data_dir    = test_data_dir,
            test_meta_path   = test_meta_path,
            model_list       = model_list,
            hyperparam_grids = hyperparam_grids,
            preprocess_grid  = preprocess_grid,
            output_dir       = out_dir,
            target_column    = col
        )
        print(f"--- {nice_name} head ---\n", loo_df.head())
        print(f"--- {nice_name} metrics ---\n", loo_metrics)
