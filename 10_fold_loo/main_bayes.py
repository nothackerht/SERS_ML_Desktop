#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_bayes.py

Implements:
1) Outer LOO over 11 external samples
2) Inner 5-fold GroupKFold with Bayesian optimization (skopt.gp_minimize) over XGBoost hyperparams
3) Fold-specific retraining & prediction (mean±std) using fold-optimal hyperparams
4) Aggregation: compute mode of the 11 fold-specific hyperparams
5) Global retraining per fold using mode hyperparams, generating new predictions
6) Save per-fold metrics (fold-specific and global) for each preprocessing chain
7) Parity plots for global predictions
Saves results to Excel/CSV and PNGs via bayes_visualizations.
"""
import os
import numpy as np
import torch
import pandas as pd
from skopt import gp_minimize
from skopt.utils import use_named_args
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from collections import Counter
from xgboost import XGBRegressor

from modules.data_loader   import load_data, load_metadata
from modules.preprocessing import Preprocessing
from modules.hyperparams   import xgb_space      # 9-dim skopt space
from modules.bayes_visualizations import plot_parity
from modules.bayes_visualizations import (
    plot_convergence_curve,
    plot_evaluations_scatter,
    plot_hyperparam_heatmap,
    plot_feature_importance,
    plot_residuals,
    plot_parity as bv_plot_parity,
    plot_shap_summary
)

# Define your preprocessing chains
preprocess_grid = [
    [], ['EMSC'], ['IntervalPLS'], ['SNV'],
    ['SNV', 'IntervalPLS'], ['Normalization', 'IntervalPLS'], ['IntervalPLS', 'Normalization'],
    ['Second Derivative'], ['EMSC', 'SNV'], ['IntervalPLS', 'EMSC', 'SNV'],
    ['EMSC', 'SNV', 'IntervalPLS'], ['SNV', 'Second Derivative'],
]

# Ensure output dirs
os.makedirs('results', exist_ok=True)
pos = 'results'

# Load external (raw‐replicate) spectra for 43 train + 11 test
def load_external(train_dir, train_meta, test_dir, test_meta):
    # raw_tr: (n_features, 9×43)=387 spectra, plus filenames fn_tr
    _, _, raw_tr, fn_tr = load_data(
        train_dir,
        metadata_path=train_meta,
        return_filenames=True
    )
    y_tr_meta = load_metadata(train_meta)['target_SI'].values

    # raw_ex: (n_features, 9×11)=99 spectra, plus filenames fn_ex
    _, _, raw_ex, fn_ex = load_data(
        test_dir,
        metadata_path=test_meta,
        return_filenames=True
    )
    y_ex_meta = load_metadata(test_meta)['target_SI'].values

    return raw_tr, y_tr_meta, fn_tr, raw_ex, y_ex_meta, fn_ex



def run_outer_loo(raw_tr, y_tr_meta, fn_tr, raw_ex, y_ex_meta, fn_ex, chain, n_calls=25):
    # number of external samples = total spectra / 9
    n_ex = raw_ex.shape[1] // 9

    # build per-spectrum target vectors and group indices
    y_tr = np.repeat(y_tr_meta, 9)    # 387 values
    y_ex = np.repeat(y_ex_meta, 9)    #  99 values
    ext_idx = np.repeat(np.arange(n_ex), 9)

    # initialize record containers
    fold_records = []
    hyper_list   = []
    
    # …and then your for-loop follows…


    # ─── Outer LOO ─────────────────────────────────────────────────────────────
    for i in range(n_ex):
        # hold‐out sample i
        hold_mask = (ext_idx == i)           # boolean mask on raw_ex columns
        X_hold    = raw_ex[:, hold_mask]     # (n_features, 9)
        y_hold    = y_ex_meta[i]             # true sample-level target

        # remaining external spectra
        rem_mask = ~hold_mask
        X_rem    = raw_ex[:, rem_mask]       # (n_features, 90)

        # pool = all train + remaining ext
        X_pool = np.hstack([raw_tr, X_rem])  
        y_pool = np.concatenate([y_tr, y_ex[rem_mask]])

        # group labels: 0..42 for train, 43.. for remaining ext
        grp_tr = np.repeat(np.arange(len(y_tr_meta)), 9)
        grp_ex = len(y_tr_meta) + ext_idx[rem_mask]
        groups = np.concatenate([grp_tr, grp_ex])

        prep = Preprocessing(X_pool)

        # ─── Inner Bayesian CV objective ────────────────────────────────────────
        @use_named_args(xgb_space)
        def objective(**params):
            # preprocess → samples × features
            Xp  = prep.preprocess(chain, y=y_pool).T
            yv  = y_pool
            gkf = GroupKFold(n_splits=5)
            rmses = []

            for tr_i, va_i in gkf.split(Xp, yv, groups):
                mdl = XGBRegressor(
                    **params,
                    tree_method = 'hist',
                    device      = 'cuda' if torch.cuda.is_available() else 'cpu',
                    random_state=42
                )
                mdl.fit(Xp[tr_i], yv[tr_i])
                preds = mdl.predict(Xp[va_i])

                # sample-level averaging
                grp_preds = {}
                for g,p in zip(groups[va_i], preds):
                    grp_preds.setdefault(g, []).append(p)
                y_pred = np.array([np.mean(v) for v in grp_preds.values()])

                # true sample-level targets
                true = np.array([
                    y_tr_meta[g] if g < len(y_tr_meta)
                    else y_ex_meta[g - len(y_tr_meta)]
                    for g in grp_preds
                ])

                rmses.append(np.sqrt(mean_squared_error(true, y_pred)))

            return float(np.mean(rmses))

        # ─── run Bayesian optimization ───────────────────────────────
        res = gp_minimize(objective, xgb_space, n_calls=n_calls, random_state=42)
        best_hp = tuple(res.x)
        hyper_list.append(best_hp)

        # save diagnostics
        plot_convergence_curve(res)
        plot_evaluations_scatter(res)
        plot_hyperparam_heatmap(res, 'max_depth', 'learning_rate')

        # ─── retrain on full pool with best_hp ───────────────────────────
        mdl_final = XGBRegressor(
            **dict(zip([d.name for d in xgb_space], best_hp)),
            tree_method = 'hist',
            device      = 'cuda' if torch.cuda.is_available() else 'cpu',
            random_state=42
        )
        X_full = prep.preprocess(chain, y=y_pool).T
        mdl_final.fit(X_full, y_pool)

        # diagnostics on fold model
        plot_feature_importance(mdl_final, top_n=20)
        plot_residuals          (mdl_final, X_full, y_pool)
        plot_shap_summary       (mdl_final, X_full)

        # ─── fold-specific prediction ─────────────────────────────────
        Xh    = Preprocessing(X_hold).preprocess(chain, y=[y_hold]).T

        preds = mdl_final.predict(Xh)
        fold_records.append({
            'fold':           i,
            'preproc':        '+'.join(chain),
            'best_hp':        best_hp,
            'fold_pred_mean': float(np.mean(preds)),
            'fold_pred_std':  float(np.std(preds)),
            'fold_rmse':      float(np.sqrt(mean_squared_error([y_hold], [preds.mean()])))
        })

    # ─── aggregate mode hyperparameters ─────────────────────────────────────
    mode_hp = Counter(hyper_list).most_common(1)[0][0]

    # ─── global retrain & per-fold prediction with mode_hp ────────────────
    global_records = []
    for i in range(n_ex):
        hold_mask = (ext_idx == i)
        X_hold    = raw_ex[:, hold_mask]
        y_hold    = y_ex_meta[i]

        rem_mask = ~hold_mask
        X_rem    = raw_ex[:, rem_mask]
        X_pool   = np.hstack([raw_tr, X_rem])
        y_pool   = np.concatenate([y_tr, y_ex[rem_mask]])

        grp_tr = np.repeat(np.arange(len(y_tr_meta)), 9)
        grp_ex = len(y_tr_meta) + ext_idx[rem_mask]
        groups = np.concatenate([grp_tr, grp_ex])

        prep = Preprocessing(X_pool)
        mdl_g = XGBRegressor(
            **dict(zip([d.name for d in xgb_space], mode_hp)),
            tree_method = 'hist',
            device      = 'cuda' if torch.cuda.is_available() else 'cpu',
            random_state=42
        )
        X_full = prep.preprocess(chain, y=y_pool).T
        mdl_g.fit(X_full, y_pool)

        Xh    = Preprocessing(X_hold).preprocess(chain, y=[y_hold]).T

        preds = mdl_g.predict(Xh)
        global_records.append({
            'fold':            i,
            'preproc':        '+'.join(chain),
            'mode_hp':         mode_hp,
            'global_pred_mean': preds.mean(),
            'global_pred_std':  preds.std(),
            'global_rmse':     np.sqrt(mean_squared_error([y_hold], [preds.mean()]))
        })

    return fold_records, global_records



# Main driver
if __name__ == '__main__':
    # ── Set your new data locations here ────────────────────────────────────────
    data_directory              = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
    meta_data_directory         = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"
    external_test_spectra_path  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data_test_updated"
    external_test_metadata_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata_test_updated.csv"
    output_dir                  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\bayes_results"

    # Load raw replicate spectra + per-sample metadata
    raw_tr, y_tr_meta, fn_tr, raw_ex, y_ex_meta, fn_ex = load_external(
        data_directory,
        meta_data_directory,
        external_test_spectra_path,
        external_test_metadata_path
    )

    # ── Main driver continued ─────────────────────────────────────────────────

    all_folds  = []
    all_global = []

    # for each preprocessing chain, run outer LOO + collect fold-specific & global records
    for chain in preprocess_grid:
        f_rec, g_rec = run_outer_loo(
            raw_tr,
            y_tr_meta,
            fn_tr,
            raw_ex,
            y_ex_meta,
            fn_ex,
            chain,
            n_calls=25
        )
        all_folds .extend(f_rec)
        all_global.extend(g_rec)

    # build and save results
    df_f   = pd.DataFrame(all_folds)
    df_g   = pd.DataFrame(all_global)
    df_both = df_f.merge(df_g, on=['fold','preproc'])
    os.makedirs(output_dir, exist_ok=True)
    df_both.to_excel(os.path.join(output_dir, 'loo_bayes_comparison.xlsx'), index=False)


    # Parity plot for the global (mode-HP) model
    y_true = df_both['fold'].apply(lambda i: y_ex_meta[i]).values
    y_pred = df_both['global_pred_mean'].values
    bv_plot_parity(y_true, y_pred)

    print("✅ Saved comparison results to", output_dir)

