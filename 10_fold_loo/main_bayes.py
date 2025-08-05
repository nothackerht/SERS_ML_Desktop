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
import pickle
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
import modules.bayes_visualizations as bvvis
import shutil

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

# # Define your preprocessing chains
# preprocess_grid = [
#     [], ['EMSC'], ['IntervalPLS'], ['SNV'],
#     ['SNV', 'IntervalPLS'], ['Normalization', 'IntervalPLS'], ['IntervalPLS', 'Normalization'],
#     ['Second Derivative'], ['EMSC', 'SNV'], ['IntervalPLS', 'EMSC', 'SNV'],
#     ['EMSC', 'SNV', 'IntervalPLS'], ['SNV', 'Second Derivative'],
# ]
# preprocess_grid = [
#     [], ['EMSC'], ['SNV'],
#     ['SNV'], ['Normalization'], ['Normalization'],
#     ['Second Derivative'], ['EMSC', 'SNV'], ['EMSC', 'SNV'],
#     ['EMSC', 'SNV',], ['SNV', 'Second Derivative'],
# ]
preprocess_grid = [
    [], ['EMSC']
]

# Ensure output dirs
os.makedirs('results', exist_ok=True)
pos = 'results'

# Load external (raw‐replicate) spectra for 43 train + 11 test
def load_external(train_dir, train_meta, test_dir, test_meta):
    # raw_tr: (n_features, 9×43)=387 spectra
    _, _, raw_tr, _ = load_data(
        train_dir,
        metadata_path=train_meta,
        return_filenames=True
    )
    train_meta_df = load_metadata(train_meta)
    y_tr_meta = train_meta_df['target_SI'].values

    # raw_ex: (n_features, 9×11)=99 spectra
    _, _, raw_ex, _ = load_data(
        test_dir,
        metadata_path=test_meta,
        return_filenames=True
    )
    test_meta_df = load_metadata(test_meta)
    y_ex_meta = test_meta_df['target_SI'].values
    sample_ids_ex = test_meta_df.index.astype(str).tolist()  # ✅ store real sample IDs

    return raw_tr, y_tr_meta, raw_ex, y_ex_meta, sample_ids_ex



def run_outer_loo(raw_tr, y_tr_meta, raw_ex, y_ex_meta, chain, n_calls=25):
    # number of external samples = total spectra / 9
    n_ex = raw_ex.shape[1] // 9

    # build per-spectrum target vectors and group indices
    y_tr = np.repeat(y_tr_meta, 9)    # 387 values
    y_ex = np.repeat(y_ex_meta, 9)    #  99 values


    # initialize record containers
    fold_records = []
    hyper_list   = []
    
    # …and then your for-loop follows…
    ext_idx = np.repeat(np.arange(raw_ex.shape[1] // 9), 9)

    # ─── Outer LOO ─────────────────────────────────────────────────────────────
    for i in range(n_ex):
        # Set per-fold subfolder output
        fold_dir = os.path.join(base_out, '+'.join(chain), f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        bvvis.OUTPUT_DIR = fold_dir

        # ─── NEW: isolate per‐fold folder ────────────────────────────────────
        
        print(f"Fold {i}: Holding out sample index {i}, ext_idx unique: {np.unique(ext_idx)}")


        
        # hold‐out sample i
        hold_mask = (ext_idx == i)
        X_hold    = raw_ex[:, hold_mask]
        y_hold    = y_ex_meta[i]

        # remaining external spectra
        rem_mask = ~hold_mask
        X_rem    = raw_ex[:, rem_mask]

        # pool = all train + remaining ext
        X_pool = np.hstack([raw_tr, X_rem])
        y_pool = np.concatenate([y_tr, y_ex[rem_mask]])
        # ✅ Add this block right here
        print(f"Fold {i}: X_pool shape = {X_pool.shape}, hash = {hash(X_pool.tobytes())}")
        print(f"Fold {i}: y_pool hash = {hash(y_pool.tobytes())}")
        # group labels
        grp_tr = np.repeat(np.arange(len(y_tr_meta)), 9)
        grp_ex = len(y_tr_meta) + ext_idx[rem_mask]
        groups = np.concatenate([grp_tr, grp_ex])

        # ✅ IntervalPLS selection — done per fold
        prep = Preprocessing(X_pool, ipls_threshold=0.2)
        sel = None
        if 'IntervalPLS' in chain:
            sel = prep.select_intervals_by_pls_r2(
                X_pool.T,
                y_pool,
                n_intervals=150,
                n_components=2,
                cv_folds=5,
                threshold=prep.ipls_threshold
            )
        if sel is not None:
            # Replace X_pool with interval-selected version BEFORE creating new Preprocessing instance
            X_pool = X_pool[sel, :]
            prep = Preprocessing(X_pool, ipls_threshold=0.2)

            print(f"Fold {i}: selected {np.sum(sel)} / 150 intervals")  # ✅ Move inside


        # ─── Inner Bayesian CV objective ────────────────────────────────────────
        @use_named_args(xgb_space)
        def objective(**params):
            gkf = GroupKFold(n_splits=5)
            rmses = []
        
            # Strip IntervalPLS since already applied
            methods_wo_ipls = [m for m in chain if m != 'IntervalPLS']



            for tr_i, va_i in gkf.split(X_pool.T, y_pool, groups):
                # Get training and validation spectra
                X_train_fold = X_pool[:, tr_i]
                y_train_fold = y_pool[tr_i]
                grp_train    = groups[tr_i]
        
                X_val_fold   = X_pool[:, va_i]
                y_val_fold   = y_pool[va_i]
                grp_val      = groups[va_i]
        
                # Fit preprocessing on train only
                fold_prep = Preprocessing(X_train_fold, ipls_threshold=0.2)
                X_train_proc = fold_prep.preprocess(methods_wo_ipls, y=y_train_fold).T
                X_val_proc   = fold_prep.preprocess(methods_wo_ipls, y=y_val_fold).T
                assert X_train_proc.shape[1] == X_val_proc.shape[1], f"Fold shape mismatch: train={X_train_proc.shape}, val={X_val_proc.shape}"

                # Fit model
                mdl = XGBRegressor(
                    **params,
                    tree_method = 'hist',
                    device      = 'cuda' if torch.cuda.is_available() else 'cpu',
                    random_state=42
                )
                mdl.fit(X_train_proc, y_train_fold)
                preds = mdl.predict(X_val_proc)
        
                # Average to sample-level
                grp_preds = {}
                for g, p in zip(grp_val, preds):
                    grp_preds.setdefault(g, []).append(p)
                y_pred = np.array([np.mean(v) for v in grp_preds.values()])
        
                # True values
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
        # ─── apply the same IntervalPLS selection to the hold-out ───────────
        if sel is not None:
            print(f"Fold {i}: Intervals selected = {np.where(sel)[0].tolist()}")
            # Apply only non-IntervalPLS preprocessing
            methods_wo_ipls = [m for m in chain if m != 'IntervalPLS']
            
            hold_prep = Preprocessing(X_hold[sel, :], ipls_threshold=0.2)
            Xh = hold_prep.preprocess(methods_wo_ipls, y=[y_hold]).T
        else:
            Xh = Preprocessing(X_hold, ipls_threshold=0.2).preprocess(chain, y=[y_hold]).T


        assert Xh.shape[1] == X_full.shape[1], f"Mismatch: test={Xh.shape}, train={X_full.shape}"
        
        preds = mdl_final.predict(Xh)
        
        # ✅ Store raw per-spectrum predictions
        y_test_spectra = np.repeat(y_hold, len(preds))  # each spectrum gets same sample-level target
        y_pred_spectra = preds.tolist()
        
        # ✅ Average to sample-level for RMSE
        avg_pred = float(np.mean(preds))
        avg_rmse = float(np.sqrt(mean_squared_error([y_hold], [avg_pred])))
        
        # Save parity plot using averaged values (keeps parity plot consistent with RMSE metric)
        bv_plot_parity(
            model_name=f"Fold_{i}_{'+'.join(chain)}",
            y_test=[y_hold],
            y_pred=[avg_pred]
        )
        
     
        fold_records.append({
            'fold':                  i,
            'held_out_sample':       sample_ids_ex[i],  # ✅ real external sample ID

            'preproc':               '+'.join(chain),
            'best_hp':               best_hp,
            'intervals':             sel,  # 🔥 store selected interval indices
            'fold_pred_mean':        avg_pred,
            'fold_pred_std':         float(np.std(preds)),
            'fold_rmse':             avg_rmse,
            'y_test_spectra':        y_test_spectra,   # 🔥 full spectral-level true values
            'y_pred_spectra':        y_pred_spectra    # 🔥 full spectral-level predictions
        })


        # Reset OUTPUT_DIR back to the main chain folder for global retraining
        bvvis.OUTPUT_DIR = os.path.join(base_out, '+'.join(chain))

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
    
        # 🔁 Look up this fold's selected intervals
        fold_sel = None
        for rec in fold_records:
            if rec['fold'] == i:
                fold_sel = rec.get('intervals', None)
                break
    

        # Apply preprocessing using correct fold_sel consistently
        if fold_sel is not None:
            methods_wo_ipls = [m for m in chain if m != 'IntervalPLS']
            
            train_prep = Preprocessing(X_pool[fold_sel, :], ipls_threshold=0.2)
            X_train_final = train_prep.preprocess(methods_wo_ipls, y=y_pool).T
        
            hold_prep = Preprocessing(X_hold[fold_sel, :], ipls_threshold=0.2)
            X_hold_final = hold_prep.preprocess(methods_wo_ipls, y=[y_hold]).T

        else:
            # Only preprocess fully if IntervalPLS wasn't part of the chain
            prep = Preprocessing(X_pool, ipls_threshold=0.2)
            X_train_final = prep.preprocess(chain, y=y_pool).T
            X_hold_final  = Preprocessing(X_hold, ipls_threshold=0.2).preprocess(chain, y=[y_hold]).T


        # 🔥 Add this assert RIGHT HERE to catch shape mismatch
        assert X_train_final.shape[1] == X_hold_final.shape[1], (
            f"🔥 Shape mismatch in fold {i}: "
            f"train shape = {X_train_final.shape}, test shape = {X_hold_final.shape}, "
            f"intervals used = {fold_sel is not None}"
        )
        # Fit and predict using mode_hp
        mdl_g = XGBRegressor(
            **dict(zip([d.name for d in xgb_space], mode_hp)),
            tree_method = 'hist',
            device      = 'cuda' if torch.cuda.is_available() else 'cpu',
            random_state=42
        )
        mdl_g.fit(X_train_final, y_pool)
        assert X_train_final.shape[1] == X_hold_final.shape[1], f"Fold {i} shape mismatch: train {X_train_final.shape}, hold {X_hold_final.shape}"

        preds = mdl_g.predict(X_hold_final)
        
        # ✅ Store raw per-spectrum predictions for debugging
        y_test_spectra = np.repeat(y_hold, len(preds))  # all 9 spectra have same target
        y_pred_spectra = preds.tolist()
        
        # ✅ Average to sample-level for RMSE
        avg_pred = float(np.mean(preds))
        avg_rmse = float(np.sqrt(mean_squared_error([y_hold], [avg_pred])))
        
        global_records.append({
            'fold':                  i,
            'held_out_sample':       sample_ids_ex[i],  # ✅ real external sample ID

            'preproc':               '+'.join(chain),
            'mode_hp':               mode_hp,
            'global_pred_mean':      avg_pred,
            'global_pred_std':       float(np.std(preds)),
            'global_rmse':           avg_rmse,
            'y_test_spectra':        y_test_spectra,    #  full spectral-level true values
            'y_pred_spectra':        y_pred_spectra     #  full spectral-level predictions
        })
        
        

    return fold_records, global_records

    

# … (your run_outer_loo and load_external definitions above) …

if __name__ == '__main__':
    # ── Set your new data locations here ────────────────────────────────────────
    data_directory              = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data"
    meta_data_directory         = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata.csv"
    external_test_spectra_path  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\data_test_updated"
    external_test_metadata_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\Data\y_metadata_test_updated.csv"
    output_dir                  = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\bayes_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load raw replicate spectra + per-sample metadata
    raw_tr, y_tr_meta, raw_ex, y_ex_meta, sample_ids_ex = load_external(
        data_directory,
        meta_data_directory,
        external_test_spectra_path,
        external_test_metadata_path
    )

    # **HERE**: Make a shortcut to the viz module and define base_out
    import modules.bayes_visualizations as bvvis
    base_out = output_dir
    # checkpoint bookkeeping
    chkpt_file = os.path.join(output_dir, 'completed_chains.pkl')
    if os.path.exists(chkpt_file):
        completed = pickle.load(open(chkpt_file, 'rb'))
    else:
        completed = []

    all_folds  = []
    all_global = []

    base_out = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\bayes_results"

    summary_dir = os.path.join(base_out, "_summary_parity_plots")
    os.makedirs(summary_dir, exist_ok=True)

    from modules import bayes_visualizations
    
    for chain in preprocess_grid:
        key = '+'.join(chain) if chain else 'none'
    
        # 1) make a subfolder for this preprocessing chain
        chain_out = os.path.join(base_out, key)
        os.makedirs(chain_out, exist_ok=True)
    
        # 2) redirect all of bayes_visualizations into that folder
        bvvis.OUTPUT_DIR = chain_out
    
        # 3) now you can do your “skip if already done” logic
        if key in completed:
            print(f"→ skipping {key}, already done")
            continue
    
        print(f"→ running chain {key} …")
        f_rec, g_rec = run_outer_loo(
            raw_tr, y_tr_meta,
            raw_ex, y_ex_meta,
            chain, n_calls=25   
        )
    
        # 4) pickle & mark done
        pickle.dump({'folds': f_rec, 'global': g_rec},
                    open(os.path.join(chain_out, f"results_{key}.pkl"), 'wb'))
        completed.append(key)
        pickle.dump(completed, open(chkpt_file, 'wb'))
    
        all_folds .extend(f_rec)
        all_global.extend(g_rec)

       

# ─── Build and save final table ──────────────────────────────────────────
df_f   = pd.DataFrame(all_folds)
df_g   = pd.DataFrame(all_global)
df_both = df_f.merge(df_g, on=['fold','preproc'])
df_both.to_excel(os.path.join(output_dir, 'loo_bayes_comparison.xlsx'), index=False)

# ─── Generate chain-wise ensemble/global parity plots ────────────────────
grouped = df_both.groupby("preproc")

for preproc, group in grouped:
    group = group.sort_values("fold")
    y_true  = group["fold"].apply(lambda i: y_ex_meta[i]).values
    y_ens   = group["fold_pred_mean"].values
    y_glob  = group["global_pred_mean"].values

    # Save both ensemble and global parity plots for this preprocessing chain
    std_ens = group["fold_pred_std"].values
    # Save in chain folder
    ens_plot = bv_plot_parity(f"Ensemble_{preproc}", y_true, y_ens, stds=std_ens)
    glob_plot = bv_plot_parity(f"Global_{preproc}", y_true, y_glob)
    
    # Copy to summary folder
    ens_src = os.path.join(bvvis.OUTPUT_DIR, f"Ensemble_{preproc}.png")
    glob_src = os.path.join(bvvis.OUTPUT_DIR, f"Global_{preproc}.png")
    
    if os.path.exists(ens_src):
        shutil.copy(ens_src, os.path.join(summary_dir, f"Ensemble_{preproc}.png"))
    if os.path.exists(glob_src):
        shutil.copy(glob_src, os.path.join(summary_dir, f"Global_{preproc}.png"))


# ─── Final overall global parity plot ─────────────────────────────────────
y_true = df_both['fold'].apply(lambda i: y_ex_meta[i]).values
y_pred = df_both['global_pred_mean'].values
bv_plot_parity("XGBoost_Global", y_true, y_pred)

print("✅ Saved comparison results to", output_dir)


