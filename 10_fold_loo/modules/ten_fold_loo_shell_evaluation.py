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
                # --- OUTER IntervalPLS feature‐selection (once per outer fold) ---
                if 'IntervalPLS' in prep_chain:
                    # X_raw is (n_spectra, n_wavenumbers) but our selector expects (n_samples, n_features)
                    X_outer = X_raw                # rows=spectra, cols=wavenumbers
                    y_outer = y_vals               # vector of length n_spectra
                    sel = Preprocessing(X_outer).select_intervals_by_pls_r2(
                        X_outer, y_outer,
                        n_intervals=150,
                        n_components=2,
                        cv_folds=5,
                        threshold=0.2
                    )
                else:
                    # no IntervalPLS → keep all features
                    sel = np.arange(X_raw.shape[1])
                # keep everything except IntervalPLS for inner‐loop & final unsupervised steps
                other_methods = [m for m in prep_chain if m != 'IntervalPLS']


               
                # Inner CV for hyperparameter tuning
                best_rmse = np.inf
                best_hp   = None
                inner     = GroupKFold(n_splits=5)
            
                for hp in hyperparam_grids.get(model_name, []):
                    fold_rmses = []
            
                    for iti, ito in inner.split(X_raw, y_vals, groups=groups):
                        # — split into train / val raw spectra & labels —
                        X_tr_raw = X_raw[iti].T      # shape (n_wavenumbers, n_train_spectra)
                        X_vl_raw = X_raw[ito].T      # shape (n_wavenumbers, n_val_spectra)
                        y_tr      = y_vals[iti]      # 1D array, length = train_spectra_count
                        y_vl      = y_vals[ito]
            
                        # — apply just the *other* (unsupervised) preprocessing on the train block —
                        prep = Preprocessing(X_tr_raw)
                        Xp_tr = prep.preprocess(other_methods)
                        
                        # — apply same unsupervised chain on the val block —
                        Xp_vl = Preprocessing(X_vl_raw).preprocess(other_methods)
                                            
                        # — now *slice* each to the selected intervals from the outer fold —
                        # (slice rows/features, not columns)
                        Xp_tr = Xp_tr[sel, :]    # keep only those feature‐rows
                        Xp_vl = Xp_vl[sel, :]
                        
                        # now Xp_tr, Xp_vl are shape (n_selected_wavenumbers, n_spectra)
                        # transpose back to (n_spectra, n_selected_wavenumbers)
                        Xp_tr = Xp_tr.T
                        Xp_vl = Xp_vl.T



            
                        # — ensure float32 + contiguous for features —
                        X_tr32 = np.ascontiguousarray(Xp_tr, dtype=np.float32)
                        X_vl32 = np.ascontiguousarray(Xp_vl, dtype=np.float32)
                        
                        # — shape y correctly for sipls vs. sklearn models —
                        if model_name == 'sipls':
                            # torch_pls wants (n_samples,1)
                            y_input = y_tr.astype(np.float32).reshape(-1, 1)
                        else:
                            # everything else (RF, SVR, XGBoost, etc.) wants (n_samples,)
                            y_input = y_tr.astype(np.float32)
                        
                        # — fit & predict —
                        mdl = get_model_by_name(model_name, hp)
                        mdl.fit(X_tr32, y_input)
                        preds = mdl.predict(X_vl32).flatten()
                        

            
                        # — compute sample-level RMSE —
                        s_true = y_vl.reshape(-1, 9).mean(axis=1)
                        s_pred = preds.reshape(-1, 9).mean(axis=1)
                        fold_rmses.append(np.sqrt(mean_squared_error(s_true, s_pred)))
            
                    avg_rmse = np.mean(fold_rmses)
                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_hp   = hp


                # ─── Retrain on full combined + predict held-out nine spectra ───
                # 1) Apply only the “other” (unsupervised) steps to the full training pool
                X_full_raw = X_raw.T  # (n_spectra_total, n_wavenumbers)
                other_methods = [m for m in prep_chain if m != 'IntervalPLS']
                prep_full = Preprocessing(X_full_raw)
                Xp_full_unsup = prep_full.preprocess(other_methods)       # (n_wavenumbers, n_spectra_total)
    
                # 2) Slice to the intervals selected in the outer fold
                Xp_full_sel = Xp_full_unsup[sel, :]                       # (n_selected_wavenumbers, n_spectra_total)
    
                # 3) Transpose to (n_spectra_total, n_selected_wavenumbers)
                Xp_full = Xp_full_sel.T                                   
    
                # 4) Cast to float32 & contiguous for training
                X_full32 = np.ascontiguousarray(Xp_full, dtype=np.float32)
    
                # 5) Prepare the full-pool target array
                if model_name == 'sipls':
                    y_full_input = y_vals.astype(np.float32).reshape(-1, 1)
                else:
                    y_full_input = y_vals.astype(np.float32)
    
                # 6) Transform the held-out block with the same unsupervised chain
                loo_raw = all_test[:, to]                                  # (n_wavenumbers, 9)
                prep_loo = Preprocessing(loo_raw)
                Xp_loo_unsup = prep_loo.preprocess(other_methods)          # (n_wavenumbers, 9)
    
                # 7) Slice to the same selected intervals
                Xp_loo_sel = Xp_loo_unsup[sel, :]                          # (n_selected_wavenumbers, 9)
    
                # 8) Transpose to (9, n_selected_wavenumbers)
                Xp_loo = Xp_loo_sel.T                                      
    
                # 9) Cast to float32 & contiguous for prediction
                X_loo32 = np.ascontiguousarray(Xp_loo, dtype=np.float32)
    
                # 5) Fit on full data & predict held-out
                mdl = get_model_by_name(model_name, best_hp)
                mdl.fit(X_full32, y_full_input)
                raw_preds = mdl.predict(X_loo32).flatten()
                # 6) Store mean/std of the nine predictions
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

    
    
    # -------------- pick best-preproc per model --------------
    # metrics is a DataFrame indexed by (model, preprocess)
    # and has an 'RMSE' column
    best_per_model = (
        metrics
        .reset_index()                                # bring model,preprocess back to columns
        .sort_values(['model','RMSE'], ascending=[True,True])
        .drop_duplicates('model', keep='first')       # keep the row with lowest RMSE per model
    )
    # now best_per_model is something like:
    #    model        preprocess       RMSE     R2
    # 0  sipls        ""               0.123   0.85
    # 1  random_forest "SNV"           0.234   0.79
    # etc.

    # -------------- plot only those --------------
    for _, row in best_per_model.iterrows():
        m    = row['model']
        prep = row['preprocess']
        rmse = row['RMSE']
        r2   = row['R2']

        # grab the predictions for that combination
        grp = df[
            (df['model'] == m) & 
            (df['preprocess'] == prep)
        ]

        y_true = grp['true'].values
        y_pred = grp['pred_mean'].values

        plt.figure(figsize=(6,6))
        plt.errorbar(
            y_true,
            y_pred,
            yerr=grp['pred_std'],
            fmt='o',
            capsize=4
        )
        mn, mx = y_true.min(), y_true.max()
        plt.plot([mn,mx], [mn,mx], 'r--')

        plt.xlabel(target_column, fontweight='bold')
        plt.ylabel(target_column, fontweight='bold')

        # put everything into the title
        plt.title(
            f"{m}\n"
            f"Preproc = {prep or 'None'}\n"
            f"Best Hyperparams: {grp['hyperparams'].mode().iloc[0]}\n"
            f"RMSE = {rmse:.3f},  R² = {r2:.3f}",
            fontsize=10,
            loc='center'
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"parity_{m.replace(' ','_')}.png"),
            dpi=300
        )
        plt.show()

    # finally return the results
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
