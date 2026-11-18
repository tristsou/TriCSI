@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
python collect_s7_tricsi_adapt.py
pause
