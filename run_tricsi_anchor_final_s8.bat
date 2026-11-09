@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
python train_tricsi_anchor_final.py --test-session S8
pause
