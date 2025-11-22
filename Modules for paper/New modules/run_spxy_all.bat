@echo on
REM Make sure we are in the right folder
cd "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\New modules"

REM ========== TEST: just print Python version so we know the bat is running ==========
python -V

REM ======================================================================
REM ========== 1) BOX 1–2 WITH CONTROLS =================================
REM ======================================================================

python SPXY_eval_no_leakage.py ^
  --data_dir  "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\data" ^
  --meta_path "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\y_metadata.csv" ^
  --include_types "DM1,Control" ^
  --out_dir   "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\SPXY\10 SPXY 1-se\SPXY Box 1-2 Controls" ^
  --results_dir_si  "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-2 with controls\target_SI" ^
  --results_dir_hgs "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-2 with controls\HGS_pp_avg" ^
  --results_dir_adf "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-2 with controls\ADF_pp_avg" ^
  --spxy_repeats 10 ^
  --random_state 42

REM ======================================================================
REM ========== 2) BOX 1–2 WITHOUT CONTROLS ===============================
REM ======================================================================

python SPXY_eval_no_leakage.py ^
  --data_dir  "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\data" ^
  --meta_path "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\y_metadata.csv" ^
  --include_types "DM1" ^
  --out_dir   "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\SPXY\10 SPXY 1-se\SPXY Box 1-2 without Controls" ^
  --results_dir_si  "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-2 without controls\target_SI" ^
  --results_dir_hgs "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-2 without controls\HGS_pp_avg" ^
  --results_dir_adf "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-2 without controls\ADF_pp_avg" ^
  --spxy_repeats 10 ^
  --random_state 42

REM ======================================================================
REM ========== 3) BOX 1–3 WITH CONTROLS =================================
REM ======================================================================

python SPXY_eval_no_leakage.py ^
  --data_dir  "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\data_combined" ^
  --meta_path "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\y_metadata_combined_in_order.csv" ^
  --include_types "DM1,Control" ^
  --out_dir   "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\SPXY\10 SPXY 1-se\SPXY Box 1-3 Controls" ^
  --results_dir_si  "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-3 with controls\target_SI" ^
  --results_dir_hgs "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-3 with controls\HGS_pp_avg" ^
  --results_dir_adf "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-3 with controls\ADF_pp_avg" ^
  --spxy_repeats 10 ^
  --random_state 42

REM ======================================================================
REM ========== 4) BOX 1–3 WITHOUT CONTROLS ===============================
REM ======================================================================

python SPXY_eval_no_leakage.py ^
  --data_dir  "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\data_combined" ^
  --meta_path "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\y_metadata_combined_in_order.csv" ^
  --include_types "DM1" ^
  --out_dir   "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\SPXY\10 SPXY 1-se\SPXY Box 1-3 without Controls" ^
  --results_dir_si  "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-3 without controls\target_SI" ^
  --results_dir_hgs "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-3 without controls\HGS_pp_avg" ^
  --results_dir_adf "C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\10-Fold CV\box 1-3 without controls\ADF_pp_avg" ^
  --spxy_repeats 10 ^
  --random_state 42

echo ======================
echo ALL SPXY RUNS FINISHED
echo ======================
pause
