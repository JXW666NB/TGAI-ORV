@echo off
set WORK=C:\Users\Administrator\orv_newbie
if not exist "%WORK%" mkdir "%WORK%"
cd /d "%WORK%"
echo ==== 1) ? PyPI ? 0.9.53??????====
C:\Python311\python.exe -m pip install -U tgai-orv==0.9.53 --no-cache-dir -q
echo ==== 2) ??????? tgai_go????? ====
C:\Python311\python.exe -c "import tgai_go, os; print('version', tgai_go.__version__); print('path', os.path.dirname(tgai_go.__file__))"
echo ==== 3) ??????????? ====
C:\Python311\python.exe -X utf8 D:\tgai_orv\smoke_min.py
echo ==== exit=%ERRORLEVEL% ====
