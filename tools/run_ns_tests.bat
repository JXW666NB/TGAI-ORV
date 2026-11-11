@echo off
cd /d D:\tgai_orv\TGAI-ORV-main
C:\Python311\python.exe -m pytest tests/test_muon_ns_numerics.py -q --tb=line
