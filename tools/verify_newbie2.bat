@echo off
set WORK=C:\Users\Administrator\orv_newbie
if not exist "%WORK%" mkdir "%WORK%"
cd /d "%WORK%"
echo ==== ????? tgai_go / ?? ====
C:\Python311\python.exe -c "import tgai_go,os;print('version',tgai_go.__version__);print('path',os.path.dirname(tgai_go.__file__))"
echo ==== ????????? PYTHONPATH ????? ====
set PYTHONPATH=D:\tgai_orv\TGAI-ORV-main\python
C:\Python311\python.exe -X utf8 D:\tgai_orv\smoke_min.py
echo ==== exit=%ERRORLEVEL% ====
