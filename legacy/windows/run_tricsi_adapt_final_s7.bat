@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
python train_tricsi_adapt_final.py --mode final --test-session S7
pause
