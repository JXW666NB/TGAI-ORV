@echo off
cd /d D:\tgai_orv\TGAI-ORV-main
C:\Python311\python.exe -m pytest tests/test_cos_sin.py tests/test_sdpa_4d.py tests/test_checkpoint_cuda.py -q --tb=short
