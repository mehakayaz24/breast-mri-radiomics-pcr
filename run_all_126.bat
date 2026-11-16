@echo off
REM Runs all 126-patient analyses one after another. Logs are saved in retrain_126\logs. Does NOT touch the 12 test patients.
cd /d C:\Users\ntech\Desktop\retrain_126
if not exist logs mkdir logs
echo [1/5] train_models.py ...
python train_models.py > logs\1_train_models.txt 2>&1
echo [2/5] train_multi_models.py ...
python train_multi_models.py > logs\2_train_multi_models.txt 2>&1
echo [3/5] train_advanced_models.py ...
python train_advanced_models.py > logs\3_train_advanced_models.txt 2>&1
echo [4/5] train_nested_tuning.py (slowest) ...
python train_nested_tuning.py > logs\4_train_nested_tuning.txt 2>&1
echo [5/5] train_final_models.py ...
python train_final_models.py > logs\5_train_final_models.txt 2>&1
echo.
echo DONE. Send me the 5 text files in C:\Users\ntech\Desktop\retrain_126\logs
pause
