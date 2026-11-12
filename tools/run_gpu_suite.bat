@echo off
cd /d D:\tgai_orv\TGAI-ORV-main
set TG_RUN_CUDA_TESTS=1
C:\Python311\python.exe -X utf8 -m pytest tests -q --tb=line
