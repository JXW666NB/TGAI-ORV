@echo off
rem PyTorch (HOPE V3 reference) harness entry. ASCII only: cmd garbles CJK in .bat
set PYTHONIOENCODING=utf-8
cd /d D:\orv_bench
C:\Python311\python.exe D:\orv_bench\q_pyt.py %*
