@echo off
cd /d D:\tgai_orv\TGAI-ORV-main
C:\Python311\python.exe -X utf8 -m pytest tests/test_attn_smem_guard.py -q --tb=short
