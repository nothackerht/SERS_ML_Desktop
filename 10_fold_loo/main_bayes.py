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

# Load external (43 train + 11 test) data
def load_external(train_dir, train_meta, test_dir, test_meta):
    _, avg_tr, _, fn_tr = load_data(train_dir, metadata_path=train_meta, return_filenames=True)
    y_tr = load_metadata(train_meta)['target_SI'].values
    _, avg_ex, _, fn_ex = load_data(test_dir, metadata_path=test_meta, return_filenames=True)
    y_ex = load_metadata(test_meta)['target_SI'].values
    return avg_tr, y_tr, fn_tr, avg_ex, y_ex, fn_ex

# Single outer-fold run: returns fold-specific and global predictions
def run_outer_loo(avg_tr, y_tr, avg_ex, y_ex, fn_ex, chain, n_calls=25):
    n_ex = avg_ex.shape[1]
    fold_records = []
    hyper_list = []  # list of tuples of best hyperparams per fold

    # Prepare static pool parts
    for i in range(n_ex):
        # Partition
        mask = np.arange(n_ex) != i
        X_hold = avg_ex[:, i:i+1]
        y_hold = y_ex[i]
        X_pool = np.hstack([avg_tr, avg_ex[:, mask]])
        y_pool = np.concatenate([y_tr, y_ex[mask]])
        # groups for 9 spectra per sample
        groups = np.concatenate([np.arange(len(y_tr)).repeat(9), np.arange(len(y_tr), len(y_tr)+len(y_ex[mask])).repeat(9)])
        prep = Preprocessing(X_pool)

        # Objective: Bayesian search over xgb_space
        @use_named_args(xgb_space)
        def objective(**params):
            Xp = prep.preprocess(chain, y=y_pool).T
            yv = np.repeat(y_pool, 1)
            gkf = GroupKFold(n_splits=5)
            rmses = []
            for tr_idx, va_idx in gkf.split(Xp, yv, groups):
                mdl = XGBRegressor(**params,
                                    tree_method='hist',
                                    device='cuda' if torch.cuda.is_available() else 'cpu',
                                    random_state=42)
                mdl.fit(Xp[tr_idx], yv[tr_idx])
                preds = mdl.predict(Xp[va_idx])
                # average per sample
                gp = {}
                for g,p in zip(groups[va_idx], preds): gp.setdefault(g,[]).append(p)
                y_pred = np.array([np.mean(v) for v in gp.values()])
                true = np.array([y_tr[g] if g < len(y_tr) else y_ex[mask][g-len(y_tr)] for g in gp.keys()])
                rmses.append(np.sqrt(mean_squared_error(true, y_pred)))
            return float(np.mean(rmses))

        # Bayesian optimization
        res = gp_minimize(objective, xgb_space, n_calls=n_calls, random_state=42)
        best_hp = tuple(res.x)
        hyper_list.append(best_hp)
        plot_convergence_curve(res)
        plot_evaluations_scatter(res)
        plot_hyperparam_heatmap(res, 'max_depth', 'learning_rate')

        # Retrain on full pool with best_hp
        mdl_final = XGBRegressor(**dict(zip([d.name for d in xgb_space], best_hp)),
                                  tree_method='hist',
                                  device='cuda' if torch.cuda.is_available() else 'cpu',
                                  random_state=42)
        X_full = prep.preprocess(chain, y=y_pool).T
        mdl_final.fit(X_full, y_pool)
        
        # ─── Diagnostics on the retrained fold‐model ───────────────────────────────
        plot_feature_importance(mdl_final, top_n=20)
        plot_residuals          (mdl_final, X_full, y_pool)
        plot_shap_summary       (mdl_final, X_full)
        
        # Fold-specific prediction
        Xh = Preprocessing(X_hold).preprocess(chain, y=[y_hold]).T
        preds = mdl_final.predict(Xh)
        fold_records.append({
            'fold': i,
            'preproc': '+'.join(chain),
            'best_hp': best_hp,
            'fold_pred_mean': float(np.mean(preds)),
            'fold_pred_std': float(np.std(preds)),
            'fold_rmse': float(np.sqrt(mean_squared_error([y_hold],[np.mean(preds)])))
        })


    # Mode hyperparams across folds
    mode_hp = Counter(hyper_list).most_common(1)[0][0]

    # Global retrain & predictions per fold using mode_hp
    global_records = []
    for i in range(n_ex):
        mask = np.arange(n_ex) != i
        X_hold = avg_ex[:, i:i+1]
        y_hold = y_ex[i]
        X_pool = np.hstack([avg_tr, avg_ex[:, mask]])
        y_pool = np.concatenate([y_tr, y_ex[mask]])
        prep = Preprocessing(X_pool)
        mdl_g = XGBRegressor(**dict(zip([d.name for d in xgb_space], mode_hp)),
                              tree_method='hist',
                              device='cuda' if torch.cuda.is_available() else 'cpu',
                              random_state=42)
        X_full = prep.preprocess(chain, y=y_pool).T
        mdl_g.fit(X_full, y_pool)
        Xh = Preprocessing(X_hold).preprocess(chain, y=[y_hold]).T
        preds = mdl_g.predict(Xh)
        global_records.append({
            'fold': i,
            'preproc': '+'.join(chain),
            'mode_hp': mode_hp,
            'global_pred_mean': float(np.mean(preds)),
            'global_pred_std': float(np.std(preds)),
            'global_rmse': float(np.sqrt(mean_squared_error([y_hold],[np.mean(preds)])))
        })

    return fold_records, global_records

# Main driver
if __name__ == '__main__':
    td, tm, _, ed, em, _ = load_external(
        train_dir = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data",
        train_meta= r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv",
        test_dir  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data_test_updated",
        test_meta = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata_test_updated.csv"
    )

    all_folds = []
    all_global = []
    for chain in preprocess_grid:
        f_rec, g_rec = run_outer_loo(td, tm, ed, em, _, chain, n_calls=25)
        all_folds.extend(f_rec)
        all_global.extend(g_rec)

    df_f = pd.DataFrame(all_folds)
    df_g = pd.DataFrame(all_global)
    df_both = df_f.merge(df_g, on=['fold','preproc'])
    df_both.to_excel(os.path.join(pos, 'loo_bayes_comparison.xlsx'), index=False)

    # Parity plot for global predictions
    y_true = df_both['fold'].apply(lambda i: em[i]).values
    y_pred = df_both['global_pred_mean'].values
    plot_parity(None, y_true, y_pred)  # use simplified parity

    print("✅ Saved comparison results and parity plot.")
