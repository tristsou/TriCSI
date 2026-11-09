@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
python collect_confirmatory_session.py --session S9
pause
