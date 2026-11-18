@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
python train_spatial_rf_s4_s6.py
pause
