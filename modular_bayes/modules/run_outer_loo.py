# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 12:58:07 2025

@author: spect
"""

import os
import numpy as np
import torch
from collections import Counter
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from modules.preprocessing import Preprocessing
from modules.bayes_visualizations import (
    plot_parity, plot_convergence_curve, plot_evaluations_scatter,
    plot_hyperparam_heatmap, plot_feature_importance, plot_residuals,
    plot_shap_summary
)
from modules.bayes_opt import optimize_xgb_with_cv

def run_outer_loo(raw_tr, y_tr_meta, raw_ex, y_ex_meta, sample_ids_ex, chain, xgb_space, base_out, n_calls=25):
    n_ex = raw_ex.shape[1] // 9
    y_tr = np.repeat(y_tr_meta, 9)
    y_ex = np.repeat(y_ex_meta, 9)
    ext_idx = np.repeat(np.arange(n_ex), 9)

    fold_records = []
    hyper_list = []

    for i in range(n_ex):
        fold_dir = os.path.join(base_out, '+'.join(chain), f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)

        hold_mask = (ext_idx == i)
        X_hold = raw_ex[:, hold_mask]
        y_hold = y_ex_meta[i]

        rem_mask = ~hold_mask
        X_rem = raw_ex[:, rem_mask]
        X_pool = np.hstack([raw_tr, X_rem])
        y_pool = np.concatenate([y_tr, y_ex[rem_mask]])

        grp_tr = np.repeat(np.arange(len(y_tr_meta)), 9)
        grp_ex = len(y_tr_meta) + ext_idx[rem_mask]
        groups = np.concatenate([grp_tr, grp_ex])

        sel = None
        prep = Preprocessing(X_pool, ipls_threshold=0.2)
        if 'IntervalPLS' in chain:
            sel = prep.select_intervals_by_pls_r2(
                X_pool.T, y_pool, n_intervals=150, n_components=2,
                cv_folds=5, threshold=prep.ipls_threshold
            )
            if sel is not None:
                X_pool = X_pool[sel, :]
                prep = Preprocessing(X_pool, ipls_threshold=0.2)

        # 🔁 Use modular Bayesian optimization here
        res = optimize_xgb_with_cv(
            X_pool, y_pool, groups, chain, xgb_space, y_tr_meta, y_ex_meta, n_calls=n_calls
        )

        best_hp = tuple(res.x)
        hyper_list.append(best_hp)
        plot_convergence_curve(res)
        plot_evaluations_scatter(res)
        plot_hyperparam_heatmap(res, 'max_depth', 'learning_rate')

        # Retrain on full pool with best hyperparameters
        mdl_final = XGBRegressor(
            **dict(zip([d.name for d in xgb_space], best_hp)),
            tree_method='hist',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_state=42
        )
        X_full = prep.preprocess(chain, y=y_pool).T
        mdl_final.fit(X_full, y_pool)

        plot_feature_importance(mdl_final, top_n=20)
        plot_residuals(mdl_final, X_full, y_pool)
        plot_shap_summary(mdl_final, X_full)

        if sel is not None:
            methods_wo_ipls = [m for m in chain if m != 'IntervalPLS']
            hold_prep = Preprocessing(X_hold[sel, :], ipls_threshold=0.2)
            Xh = hold_prep.preprocess(methods_wo_ipls, y=[y_hold]).T
        else:
            Xh = Preprocessing(X_hold, ipls_threshold=0.2).preprocess(chain, y=[y_hold]).T

        preds = mdl_final.predict(Xh)
        y_test_spectra = np.repeat(y_hold, len(preds))
        y_pred_spectra = preds.tolist()
        avg_pred = float(np.mean(preds))
        avg_rmse = float(np.sqrt(mean_squared_error([y_hold], [avg_pred])))

        plot_parity(f"Fold_{i}_{'+'.join(chain)}", [y_hold], [avg_pred])

        fold_records.append({
            'fold': i,
            'held_out_sample': sample_ids_ex[i],
            'preproc': '+'.join(chain),
            'best_hp': best_hp,
            'intervals': sel,
            'fold_pred_mean': avg_pred,
            'fold_pred_std': float(np.std(preds)),
            'fold_rmse': avg_rmse,
            'y_test_spectra': y_test_spectra,
            'y_pred_spectra': y_pred_spectra
        })

    mode_hp = Counter(hyper_list).most_common(1)[0][0]
    global_records = []

    for i in range(n_ex):
        hold_mask = (ext_idx == i)
        X_hold = raw_ex[:, hold_mask]
        y_hold = y_ex_meta[i]

        rem_mask = ~hold_mask
        X_rem = raw_ex[:, rem_mask]
        X_pool = np.hstack([raw_tr, X_rem])
        y_pool = np.concatenate([y_tr, y_ex[rem_mask]])

        grp_tr = np.repeat(np.arange(len(y_tr_meta)), 9)
        grp_ex = len(y_tr_meta) + ext_idx[rem_mask]
        groups = np.concatenate([grp_tr, grp_ex])

        fold_sel = None
        for rec in fold_records:
            if rec['fold'] == i:
                fold_sel = rec.get('intervals', None)
                break

        if fold_sel is not None:
            methods_wo_ipls = [m for m in chain if m != 'IntervalPLS']
            train_prep = Preprocessing(X_pool[fold_sel, :], ipls_threshold=0.2)
            X_train_final = train_prep.preprocess(methods_wo_ipls, y=y_pool).T
            hold_prep = Preprocessing(X_hold[fold_sel, :], ipls_threshold=0.2)
            X_hold_final = hold_prep.preprocess(methods_wo_ipls, y=[y_hold]).T
        else:
            prep = Preprocessing(X_pool, ipls_threshold=0.2)
            X_train_final = prep.preprocess(chain, y=y_pool).T
            X_hold_final = Preprocessing(X_hold, ipls_threshold=0.2).preprocess(chain, y=[y_hold]).T

        assert X_train_final.shape[1] == X_hold_final.shape[1]

        mdl_g = XGBRegressor(
            **dict(zip([d.name for d in xgb_space], mode_hp)),
            tree_method='hist',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_state=42
        )
        mdl_g.fit(X_train_final, y_pool)
        preds = mdl_g.predict(X_hold_final)
        avg_pred = float(np.mean(preds))
        avg_rmse = float(np.sqrt(mean_squared_error([y_hold], [avg_pred])))

        global_records.append({
            'fold': i,
            'held_out_sample': sample_ids_ex[i],
            'preproc': '+'.join(chain),
            'mode_hp': mode_hp,
            'global_pred_mean': avg_pred,
            'global_pred_std': float(np.std(preds)),
            'global_rmse': avg_rmse,
            'y_test_spectra': np.repeat(y_hold, len(preds)),
            'y_pred_spectra': preds.tolist()
        })

    return fold_records, global_records
