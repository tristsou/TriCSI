@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
python collect_s8_anchor_final.py
pause
