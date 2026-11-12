@echo off
set WORK=C:\Users\Administrator\orv_wheel
if exist "%WORK%" rmdir /s /q "%WORK%"
mkdir "%WORK%"
cd /d "%WORK%"
echo ==== ????? wheel ????????????? pip ??====
C:\Python311\python.exe -m pip install --target "%WORK%\site" --no-deps -q D:\tgai_orv\giveme.whl
echo ==== ??????? PYTHONPATH ? ====
set PYTHONPATH=%WORK%\site
C:\Python311\python.exe -c "import tgai_go;print('version',tgai_go.__version__)"
C:\Python311\python.exe -X utf8 D:\tgai_orv\smoke_min.py
echo ==== exit=%ERRORLEVEL% ====
