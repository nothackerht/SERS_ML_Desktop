# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:59:02 2025

@author: spect

This script adds SPXY splitting for SiPLS hyperparameter optimization
and then expands the final best intervals by up to ±50 points,
retraining on the full training set and evaluating on a blind test set.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from modules.torch_pls import TorchPLS

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from modules.preprocessing import Preprocessing
from modules.data_loader import load_data, load_metadata
from itertools import product, combinations
from joblib import Parallel, delayed



def spxy_split(X, y, ratio=0.8):
    """
    SPXY split, preserving samples (9 spectra per sample).
    X: (n_spectra, n_wavenumbers), y: (n_spectra, n_targets)
    Returns train sample indices and val sample indices.
    """
    num_samples = X.shape[0] // 9
    X_avg = np.mean(X.reshape(num_samples, 9, -1), axis=1)
    y_avg = np.mean(y.reshape(num_samples, 9, -1), axis=1)
    D_X = cdist(X_avg, X_avg, 'euclidean')
    D_y = cdist(y_avg, y_avg, 'euclidean')
    D = D_X + D_y

    selected = [np.unravel_index(np.argmax(D), D.shape)[0]]
    while len(selected) < int(num_samples * ratio):
        remaining = list(set(range(num_samples)) - set(selected))
        dist_to_sel = [min(D[i, selected]) for i in remaining]
        selected.append(remaining[np.argmax(dist_to_sel)])
    train_idx = np.array(selected)
    val_idx = np.setdiff1d(np.arange(num_samples), train_idx)
    return train_idx, val_idx

def shift_best_combo_only(best_combo, max_shift=50, max_wavenumber_idx=None):
    """
    Expand each interval in best_combo by ±max_shift points.
    best_combo: list of (start,end) tuples.
    """
    shifted = []
    for pattern in product(range(-max_shift, max_shift+1), repeat=2*len(best_combo)):
        combo = []
        ok = True
       


        for i, (s, e) in enumerate(best_combo):
            ns = max(0, s + pattern[2*i])
            ne = min(max_wavenumber_idx, e + pattern[2*i+1]) if max_wavenumber_idx is not None else e + pattern[2*i+1]
            if ne <= ns:
                ok = False
                break
            combo.append((ns, ne))
        if ok:
            shifted.append(combo)
    return shifted

class RegressionModel:
    def __init__(self, all_spectra, y_label_df, wavenumbers=None):
        self.all_spectra = all_spectra
        self.y_label_df = y_label_df
        self.wavenumbers = wavenumbers

    def preprocess_and_filter_data(self):
        """
        Load raw spectra and metadata, filter to Control/DM1, assign targets,
        and return X_raw (wvn, spectra) and y_full (spectra, targets).
        """
        X_raw = self.all_spectra.copy()
        df = self.y_label_df[self.y_label_df['Type'].isin(['Control','DM1'])].reset_index(drop=True)
        df.loc[df['Type']=='Control','target_SI'] = 0
        df.loc[df['Type']=='Control',['HGS_pp_avg','ADF_pp_avg']] = 100
        valid = df.dropna(subset=['ADF_pp_avg']).index
        df = df.loc[valid].reset_index(drop=True)

        y = np.vstack([df['target_SI'], df['HGS_pp_avg'], df['ADF_pp_avg']]).T
        idxs = np.concatenate([np.arange(i*9, (i+1)*9) for i in valid])
        X = X_raw[:, idxs]
        y_rep = np.repeat(y, 9, axis=0)
        return X, y_rep

    # Inside your RegressionModel class
    
    def sipls_single_run_with_log(self,
            X_train, y_train, X_test, y_test,
            max_components=10,
            num_intervals=10,
            max_combo_intervals=4,
            wavenumbers=None,
            top_k_intervals=30):  # NEW PARAM: top-k filter
    
        splits    = np.linspace(0, X_train.shape[1], num_intervals+1, dtype=int)
        all_intervals = [(splits[i], splits[i+1]) for i in range(num_intervals)]
        groups    = np.repeat(np.arange(X_train.shape[0]//9), 9)
        gkf       = GroupKFold(n_splits=5)
    
        # STEP 1: Evaluate each individual interval
        interval_scores = []
        for (s, e) in all_intervals:
            cols = np.arange(s, e)
            r2s = []
            for train_idx, val_idx in gkf.split(X_train[:, cols], y_train, groups=groups):
                Xtr, Xva = X_train[train_idx][:, cols], X_train[val_idx][:, cols]
                ytr, yva = y_train[train_idx], y_train[val_idx]
    
                pls = TorchPLS(n_components=max_components, device='cuda')
                pls.fit(Xtr, ytr)
                pred = pls.predict(Xva)
                r2 = r2_score(yva, pred, multioutput='raw_values').mean()
                r2s.append(r2)
            interval_scores.append((s, e, np.mean(r2s)))
    
        # STEP 2: Keep top-k intervals by average R2
        top_intervals = sorted(interval_scores, key=lambda x: x[2], reverse=True)[:top_k_intervals]
        # OPTIONAL: Log top-k individual intervals to Excel
        if wavenumbers is not None:
            interval_log = pd.DataFrame({
                "Start Index": [s for s, e, _ in top_intervals],
                "End Index":   [e for s, e, _ in top_intervals],
                "Start cm⁻¹":  [wavenumbers[s] for s, e, _ in top_intervals],
                "End cm⁻¹":    [wavenumbers[e-1] for s, e, _ in top_intervals],
                "Mean R²":     [r2 for _, _, r2 in top_intervals]
            })
        else:
            interval_log = pd.DataFrame({
                "Start Index": [s for s, e, _ in top_intervals],
                "End Index":   [e for s, e, _ in top_intervals],
                "Mean R²":     [r2 for _, _, r2 in top_intervals]
            })
        
        log_path = os.path.join("TopK_Interval_Ranking.xlsx")
        interval_log.to_excel(log_path, index=False)
        print(f"Top-{top_k_intervals} interval R² scores saved to: {log_path}")

        filtered_intervals = [(s, e) for s, e, _ in top_intervals]
    
        # STEP 3: Proceed with combinations as before
        best_r2 = -np.inf
        best_combo = None
        inner_logs = []
        combo_logs = []
    
        for k in range(1, max_combo_intervals+1):
            for combo in combinations(filtered_intervals, k):
                cols = np.hstack([np.arange(s,e) for (s,e) in combo])
                r2s  = []
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train[:,cols], y_train, groups=groups), 1):
                    Xtr, Xva = X_train[tr_idx][:,cols], X_train[va_idx][:,cols]
                    ytr, yva = y_train[tr_idx],        y_train[va_idx]
    
                    pls = TorchPLS(n_components=max_components, device='cuda')
                    pls.fit(Xtr, ytr)
                    yva_pred = pls.predict(Xva)
                    fold_r2  = r2_score(yva, yva_pred, multioutput='raw_values').mean()
                    r2s.append(fold_r2)
    
                    inner_logs.append({
                        'Combo': combo,
                        'Fold': fold,
                        'Fold_R2': fold_r2,
                        'TrainIdx': tr_idx.tolist(),
                        'ValIdx':   va_idx.tolist()
                    })
    
                avg_r2 = np.mean(r2s)
                combo_logs.append({
                    'Combo': combo,
                    'AvgInnerR2': avg_r2,
                    'Start_cm⁻¹': [wavenumbers[s] for (s,_) in combo] if wavenumbers is not None else None,
                    'End_cm⁻¹':   [wavenumbers[e-1] for (_,e) in combo] if wavenumbers is not None else None,
                    'NumIntervals': k
                })
    
                if avg_r2 > best_r2:
                    best_r2    = avg_r2
                    best_combo = combo
    
        best_cols = np.hstack([np.arange(s,e) for (s,e) in best_combo])
        pls_full = TorchPLS(n_components=max_components, device='cuda')
        pls_full.fit(X_train[:,best_cols], y_train)
        y_pred = pls_full.predict(X_test[:,best_cols])
    
        return y_pred, best_combo, inner_logs, combo_logs
    def _eval_combo(self, X_tr, y_tr, X_va, y_va, C, I):
        try:
            y_val_pred, combo, *_ = self.sipls_single_run_with_log(
                X_tr, y_tr, X_va, y_va,
                max_components=C,
                num_intervals=I,
                max_combo_intervals=4,
                wavenumbers=self.wavenumbers
            )
            rmse_val = np.sqrt(mean_squared_error(y_va, y_val_pred))
            r2_val   = r2_score(y_va, y_val_pred, multioutput='raw_values').mean()
            return {'C': C, 'I': I, 'val_RMSE': rmse_val, 'val_R2': r2_val, 'combo': combo}
        except Exception as e:
            print(f"Failed for C={C}, I={I} → {e}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_test_set_with_sipls_spxy(
        self,
        test_spectra_path,
        test_metadata_path,
        preprocess_methods_list=None,
        n_components_list=[2,3,4,5,7,8,9],
        num_intervals_list=[10,15,20,30,40,50,60,70,80,90,100,110,120,130,140,150],
        max_shift=50,
        output_dir="."
    ):
        """
        For EACH pretreatment in preprocess_methods_list (and None):
          1) SPXY split (80/20)
          2) Preprocess train, val, test
          3) Grid-search C & I on SPXY-val
          4) Expand best interval ±max_shift
          5) Retrain on ALL SPXY data
          6) Blind‐test evaluation
          7) Save a summary Excel + parity plot in a subfolder
        """
        import os
    

        # 0) set up list of pretreatments
        if preprocess_methods_list is None:
            preprocess_methods_list = [
                [],                     # no preprocessing
                ['EMSC'],
                ['Normalization'],
                ['SNV'],
                ['Second Derivative']
            ]
        all_summaries = []

        # loop over each pretreatment
        for methods in preprocess_methods_list:
            # create a friendly name for this pretreatment
            name = 'none' if not methods else '_'.join(m.lower().replace(' ', '') for m in methods)
            out_dir = os.path.join(output_dir, name)
            os.makedirs(out_dir, exist_ok=True)

            # ── 1) SPXY split ─────────────────────────────────────────────────────────
            X_raw, y_full = self.preprocess_and_filter_data()
            X_all   = X_raw.T
            samples = np.repeat(np.arange(y_full.shape[0]//9), 9)
            tr_samps, va_samps = spxy_split(X_all, y_full)
            mask_tr = np.isin(samples, tr_samps)
            mask_va = np.isin(samples, va_samps)
            X_tr_raw, y_tr = X_all[mask_tr], y_full[mask_tr]
            X_va_raw, y_va = X_all[mask_va], y_full[mask_va]

            # ── 2) Preprocess splits & test ───────────────────────────────────────────
            # Apply baseline correction to SPXY splits
            X_tr = Preprocessing(X_tr_raw.T).baseline_correction().T
            X_tr = Preprocessing(X_tr.T).preprocess(methods).T
            
            X_va = Preprocessing(X_va_raw.T).baseline_correction().T
            X_va = Preprocessing(X_va.T).preprocess(methods).T


      
            # if EMSC is in here, precompute ref
            if 'EMSC' in methods:
                ref = X_tr.mean(axis=0)

            for m in methods:
                if m == 'EMSC':
                    X_tr = Preprocessing(None).emsc(X_tr.T, reference=ref).T
                    X_va = Preprocessing(None).emsc(X_va.T, reference=ref).T
                else:
                    X_tr = Preprocessing(X_tr.T).baseline_correction().T
                    X_tr = Preprocessing(X_tr.T).preprocess([m]).T
                    X_va = Preprocessing(X_va.T).baseline_correction().T
                    X_va = Preprocessing(X_va.T).preprocess([m]).T

            # also preprocess blind test later, so remember methods/ref


            # fire off all (C,I) combinations in parallel
            all_results = Parallel(n_jobs=-1, verbose=10)(
                delayed(eval_combo_wrapper)(
                    X_tr, y_tr, X_va, y_va, C, I,
                    self.wavenumbers, self.sipls_single_run_with_log
                )
                for C in n_components_list
                for I in num_intervals_list
            )


            
            # drop any failures
            grid = [r for r in all_results if r is not None]
            
            # select the best C,I by validation RMSE
            best = min(grid, key=lambda x: x['val_RMSE'])
            # ── Plot: Validation RMSE vs. # Components ───────────────────────────────
            rmse_per_c = {}
            for r in grid:
                c = r['C']
                if c not in rmse_per_c:
                    rmse_per_c[c] = []
                rmse_per_c[c].append(r['val_RMSE'])
            
            avg_rmse_c = {c: np.mean(rmses) for c, rmses in rmse_per_c.items()}
            sorted_c = sorted(avg_rmse_c.keys())
            sorted_rmse = [avg_rmse_c[c] for c in sorted_c]
            
            plt.figure(figsize=(6,4))
            plt.plot(sorted_c, sorted_rmse, marker='o')
            plt.xlabel("Number of Components", fontsize=12, fontweight='bold')
            plt.ylabel("Avg Validation RMSE", fontsize=12, fontweight='bold')
            plt.title(f"{name} | Validation RMSE vs. Components", fontsize=13)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "RMSE_vs_Components.png"), dpi=300)
            plt.show()

            best_rmse, best_r2 = best['val_RMSE'], best['val_R2']
            C_opt, I_opt, best_combo = best['C'], best['I'], best['combo']

            # ── 4) Retrain + validate on original best_combo ─────────────────────────
            # best_combo might be a single (s,e) tuple, or a tuple of k intervals
            if isinstance(best_combo[0], (int, np.integer)):
                intervals = [best_combo]              # single interval
            else:
                intervals = list(best_combo)         # list of intervals
            
            # build the combined mask of all selected intervals
            cols = np.hstack([ np.arange(s, e) for s, e in intervals ])
            
            # fit on that union of columns
            pls_cal = TorchPLS(n_components=C_opt, device='cuda')

            pls_cal.fit(X_tr[:, cols], y_tr)
            
            # predict & score on the training split
            y_tr_pred  = pls_cal.predict(X_tr[:, cols])
            train_r2   = r2_score(y_tr, y_tr_pred, multioutput='raw_values')
            train_rmse = np.sqrt(mean_squared_error(y_tr, y_tr_pred))

            # ── 5) Sequentially expand ±max_shift ───────────────────────────────────────
            # --- turn best_combo into a mutable list of intervals
            if isinstance(best_combo, tuple):
                combo_list = [best_combo]
            else:
                combo_list = list(best_combo)
            
            max_idx = len(self.wavenumbers) - 1
            current_best_rmse = best_rmse   # start from your validation RMSE on the original best_combo
            final_combo = combo_list.copy()
            # --- Normalize final_combo ---
            if isinstance(final_combo, (list, np.ndarray)):
                # Unwrap if it's a list with one element and that element is a tuple/list of intervals
                if len(final_combo) == 1 and isinstance(final_combo[0], (list, tuple)) and all(isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2 for x in final_combo[0]):
                    final_combo = list(map(tuple, final_combo[0]))  # unwrap one level
                elif all(isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2 for x in final_combo):
                    final_combo = list(map(tuple, final_combo))     # already good format
                else:
                    raise ValueError(f"Unexpected final_combo format: {final_combo}")
            else:
                raise ValueError(f"Unexpected final_combo type: {type(final_combo)}")

            final_combo = list(map(tuple, final_combo))  # normalize format
            interval_shift_log = []  # NEW: to log shifts

            # for each interval in turn, try shifting just that one
            for i, (s, e) in enumerate(final_combo):
                best_s, best_e = s, e
                for delta_s in range(-max_shift, max_shift+1):
                    for delta_e in range(-max_shift, max_shift+1):
                        ns = max(0, s + delta_s)
                        ne = min(max_idx, e + delta_e)
                        if ne <= ns:
                            continue
                        # build a trial combo where only interval i is shifted
                        trial_combo = final_combo.copy()
                        trial_combo[i] = (ns, ne)
                        # collect all columns from every interval in trial_combo
                        cols = np.hstack([np.arange(a, b) for (a, b) in trial_combo])
                        # fit & evaluate
                        pls = TorchPLS(n_components=C_opt, device='cuda')


                        pls.fit(X_tr[:, cols], y_tr)
                        yv = pls.predict(X_va[:, cols])
                        rm = np.sqrt(mean_squared_error(y_va, yv))
                        if rm < current_best_rmse:
                            current_best_rmse = rm
                            best_s, best_e = ns, ne
                # fix interval i to its locally‐best shifted bounds
                final_combo[i] = (best_s, best_e)
                # Log the shift for this interval
                delta_start = best_s - s
                delta_end   = best_e - e
                interval_shift_log.append({
                    "Interval Index": i,
                    "Original Start Index": s,
                    "Original End Index": e,
                    "Shifted Start Index": best_s,
                    "Shifted End Index": best_e,
                    "Δ Start": delta_start,
                    "Δ End": delta_end,
                    "Original Start cm⁻¹": self.wavenumbers[s],
                    "Original End cm⁻¹": self.wavenumbers[e-1],
                    "Shifted Start cm⁻¹": self.wavenumbers[best_s],
                    "Shifted End cm⁻¹": self.wavenumbers[best_e-1]
                })


           
            
            # Save interval shift log to Excel
            interval_log_df = pd.DataFrame(interval_shift_log)
            interval_log_path = os.path.join(out_dir, "Interval_Shift_Log.xlsx")
            interval_log_df.to_excel(interval_log_path, index=False)
            print(f"📄 Interval shift log saved to:\n{interval_log_path}")


            # ── 6) Retrain on ALL SPXY data ─────────────────────────────────────────
            # build columns from your final, sequentially‐shifted combo
            cols_full = np.hstack([np.arange(a, b) for (a, b) in final_combo])
            pls_full = TorchPLS(n_components=C_opt, device='cuda')




            pls_full.fit(X_all[:, cols_full], y_full)
            
            # compute retraining metrics
            y_all_pred   = pls_full.predict(X_all[:, cols_full])
            retrain_r2   = r2_score(y_full, y_all_pred, multioutput='raw_values')
            retrain_rmse = np.sqrt(mean_squared_error(y_full, y_all_pred))
            
            # … after retraining on all SPXY data …
            self.plot_training_spectra_with_intervals(
                wavenumbers=self.wavenumbers,
                X_train=X_tr,
                intervals=final_combo,
                output_path=os.path.join(out_dir, 'Train_Spectra_with_Intervals.png'),
                preprocessing_name=name,
                n_components=C_opt,
                n_intervals=I_opt
            )
            
            # ── 7) Blind-test ────────────────────────────────────────────────────────
            wavs, _, test_raw = load_data(test_spectra_path)
            test_meta         = load_metadata(test_metadata_path)
            X_test            = test_raw.T.copy()
            X_test            = Preprocessing(X_test.T).baseline_correction().T
            
                        # Predict all 9 spectra
            y_spec   = pls_full.predict(X_test[:,cols_full])
            n_test   = y_spec.shape[0]//9
            
            # Reshape and compute mean and std across the 9 spectra per sample
            reshaped_preds = y_spec.reshape(n_test, 9, -1)
            avg_pred_si = reshaped_preds.mean(axis=1)[:, 0]
            std_pred_si = reshaped_preds.std(axis=1)[:, 0]
            
            # Get true SI values and mask for valid (non-NaN) ones
            true_si = test_meta['target_SI'].values[:n_test]
            mask = ~np.isnan(true_si)
            
            # Filter both predictions and error bars using the mask
            true_si = true_si[mask]
            avg_pred_si = avg_pred_si[mask]
            std_pred_si = std_pred_si[mask]

            # Save per-sample predictions for this preprocessing
            pred_df = pd.DataFrame({
                'Preprocessing': name,
                'Sample_Index': np.arange(len(true_si)),
                'True_SI': true_si,
                'Predicted_SI': avg_pred_si,
                'Std_Dev': std_pred_si
            })

            
            # Append to full list
            if 'all_preds' not in locals():
                all_preds = pred_df
            else:
                all_preds = pd.concat([all_preds, pred_df], ignore_index=True)

            rmse_test = np.sqrt(mean_squared_error(true_si, avg_pred_si))
            r2_test   = r2_score(true_si, avg_pred_si)

            # ── parity plot & summary ───────────────────────────────────────────────
            summary = pd.DataFrame([
                {'Set':'Calibration',     'RMSE':train_rmse,   'R2':train_r2.mean()},
                {'Set':'SPXY-Validation', 'RMSE':best_rmse,    'R2':best_r2},
                {'Set':'Retrain',         'RMSE':retrain_rmse, 'R2':retrain_r2.mean()},
                {'Set':'Blind test',      'RMSE':rmse_test,     'R2':r2_test},
            ])

            summary.to_excel(os.path.join(out_dir,'AllMetrics.xlsx'), index=False)
            summary['Preprocessing'] = name  # Tag with current method
            all_summaries.append(summary)

            plt.figure(figsize=(6,6))
            plt.errorbar(
                true_si, avg_pred_si,
                yerr=std_pred_si,  # Or yerr=std_pred_si * 5 for debugging
                fmt='o',
                ecolor='gray',
                capsize=4,
                capthick=1,
                elinewidth=1,
                markeredgecolor='k',
                markersize=6
            )
            mn, mx = true_si.min(), true_si.max()
            plt.plot([mn, mx], [mn, mx], 'r--')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.xlabel('Actual SI', fontsize=12, fontweight='bold')
            plt.ylabel('Predicted SI', fontsize=12, fontweight='bold')
            plt.title(f"{name} | C={C_opt} I={I_opt} | R²={r2_test:.3f}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'Parity.png'), dpi=300)
            plt.show()

        # Save combined summary after all preprocessing methods
        combined_summary_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SiPLS SPXY\SPXY RESULTS\Combined_Model_Performance.xlsx"
        combined_df = pd.concat(all_summaries, ignore_index=True)
        combined_df.to_excel(combined_summary_path, index=False)
        print(f"✅ Combined model performance saved to:\n{combined_summary_path}")
        # Save all predictions to Excel
        preds_path = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SiPLS SPXY\SPXY RESULTS\All_Predictions_PerSample.xlsx"
        all_preds.to_excel(preds_path, index=False)
        print(f"📄 Per-sample predictions saved to:\n{preds_path}")
        


    def plot_selected_sipls_intervals(self, selected_intervals, output_dir, title,
                                      save_plot=True, show_plot=True):
        if self.wavenumbers is None:
            return
        fig, ax = plt.subplots(figsize=(10,2.5))
        for (s, e) in selected_intervals:
            ax.axvspan(self.wavenumbers[s], self.wavenumbers[e-1],
                       color='skyblue', alpha=0.5)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity")
        ax.set_title(title)
        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(output_dir,f"{title.replace(' ','_')}.png"), dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_training_spectra_with_intervals(
        self,
        wavenumbers: np.ndarray,
        X_train: np.ndarray,
        intervals: list[tuple[int,int]],
        output_path: str | None = None,
        preprocessing_name: str = "",
        n_components: int = None,
        n_intervals: int = None
    ):

        """
        Overlay all training spectra and then shade the selected intervals.
        """
        # ensure X_train is (n_spectra, n_wavenumbers)
        if X_train.shape[1] != wavenumbers.size and X_train.shape[0] == wavenumbers.size:
            spec = X_train.T
        else:
            spec = X_train
    
        fig, ax = plt.subplots(figsize=(10, 4))
        # plot every single spectrum
        for spectrum in spec:
            ax.plot(wavenumbers, spectrum,
                    color='blue', alpha=0.1, linewidth=0.5)
    
        # now shade each interval on top
        for (s, e) in intervals:
            ax.axvspan(
                wavenumbers[s], wavenumbers[e-1],
                color='skyblue', alpha=0.4
            )
    
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Intensity (A.U.)",   fontsize=14, fontweight='bold')
        # Format interval wavenumber ranges
        interval_labels = ', '.join(
            f"{self.wavenumbers[s]:.0f}-{self.wavenumbers[e-1]:.0f}" for (s, e) in intervals
        )
        
        title = (
            f"{preprocessing_name} | C={n_components}, I={n_intervals} | "
            f"{len(intervals)} Intervals: {interval_labels}"
        )
        
        ax.set_title(title, fontsize=12, fontweight='bold')

    
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
def eval_combo_wrapper(X_tr, y_tr, X_va, y_va, C, I, wavenumbers, sipls_func):
    try:
        y_val_pred, combo, *_ = sipls_func(
            X_tr, y_tr, X_va, y_va,
            max_components=C,
            num_intervals=I,
            max_combo_intervals=4,
            wavenumbers=wavenumbers
        )
        rmse_val = np.sqrt(mean_squared_error(y_va, y_val_pred))
        r2_val   = r2_score(y_va, y_val_pred, multioutput='raw_values').mean()
        return {'C': C, 'I': I, 'val_RMSE': rmse_val, 'val_R2': r2_val, 'combo': combo}
    except Exception as e:
        print(f"Failed for C={C}, I={I} → {e}")
        import traceback
        traceback.print_exc()
        return None
