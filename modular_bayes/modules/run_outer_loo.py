# Cleaned run_outer_loo() without IntervalPLS
# Only supports standard preprocessing

import os
import numpy as np
import torch
from collections import Counter
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
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

        prep = Preprocessing()
        X_proc = prep.fit_transform(X_pool, chain, y=y_pool).T

        res = optimize_xgb_with_cv(
            X_pool, y_pool, groups, chain, xgb_space, y_tr_meta, y_ex_meta, n_calls=n_calls
        )

        best_hp = tuple(res.x)
        hyper_list.append(best_hp)
        plot_convergence_curve(res)
        plot_evaluations_scatter(res)
        plot_hyperparam_heatmap(res, 'max_depth', 'learning_rate')

        mdl_final = XGBRegressor(
            **dict(zip([d.name for d in xgb_space], best_hp)),
            tree_method='hist',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_state=42
        )
        mdl_final.fit(X_proc, y_pool)
        joblib.dump(mdl_final, os.path.join(fold_dir, 'mdl_final.pkl'))

        plot_feature_importance(mdl_final, top_n=20)
        plot_residuals(mdl_final, X_proc, y_pool)
        plot_shap_summary(mdl_final, X_proc)

        hold_prep = Preprocessing()
        Xh = hold_prep.fit_transform(X_hold, chain, y=[y_hold]).T

        preds = mdl_final.predict(Xh)
        avg_pred = float(np.mean(preds))
        avg_rmse = float(np.sqrt(mean_squared_error([y_hold], [avg_pred])))

        plot_parity(f"Fold_{i}_{'+'.join(chain)}", [y_hold], [avg_pred])

        fold_records.append({
            'fold': i,
            'held_out_sample': sample_ids_ex[i],
            'preproc': '+'.join(chain),
            'best_hp': best_hp,
            'intervals': None,
            'fold_pred_mean': avg_pred,
            'fold_pred_std': float(np.std(preds)),
            'fold_rmse': avg_rmse,
            'y_test_spectra': np.repeat(y_hold, len(preds)),
            'y_pred_spectra': preds.tolist()
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

        prep = Preprocessing()
        X_train_final = prep.fit_transform(X_pool, chain, y_pool).T
        X_hold_final = prep.fit_transform(X_hold, chain, y=[y_hold]).T

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

    X_all = np.hstack([raw_tr, raw_ex])
    y_all = np.concatenate([np.repeat(y_tr_meta, 9), np.repeat(y_ex_meta, 9)])

    prep = Preprocessing()
    X_final = prep.fit_transform(X_all, chain, y_all).T

    final_model = XGBRegressor(
        **dict(zip([d.name for d in xgb_space], mode_hp)),
        tree_method='hist',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        random_state=42
    )
    final_model.fit(X_final, y_all)

    final_model_path = os.path.join(base_out, '+'.join(chain), 'final_global_model.pkl')
    joblib.dump(final_model, final_model_path)
    print(f"✅ Saved final global model to: {final_model_path}")

    return fold_records, global_records
