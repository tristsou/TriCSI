@echo off
cd /d %~dp0
py -3.14 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import sys,numpy,matplotlib,serial,sklearn,joblib; print(sys.version); print('TriCSI environment OK')"
pause
