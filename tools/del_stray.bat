@echo off
del /f /q "D:\tgai_orv\TGAI-ORV-main\tools\h2h_clean.json" 2>nul
if exist "D:\tgai_orv\TGAI-ORV-main\tools\h2h_clean.json" (echo STILL_THERE) else (echo GONE)
cd /d D:\tgai_orv\TGAI-ORV-main
C:\Python311\python.exe -X utf8 -m pytest tests/test_hygiene.py -q --tb=line
