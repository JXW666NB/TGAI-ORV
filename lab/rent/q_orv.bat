@echo off
rem ORV entry: package v3_pretrain.main() + injected tie=True (ORV_TIE=1). ASCII only.
set PYTHONIOENCODING=utf-8
set ORV_TIE=1
cd /d D:\orv_bench
D:\orv_pip_0962\Scripts\python.exe D:\orv_bench\tie_main.py %*
