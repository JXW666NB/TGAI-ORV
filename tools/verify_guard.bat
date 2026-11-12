@echo off
cd /d D:\tgai_orv\TGAI-ORV-main
echo ==== ???? ====
C:\Python311\python.exe -X utf8 -m pytest tests/test_attn_smem_guard.py -q --tb=short
echo ==== dh=64 ?????????????????====
cd tools
C:\Python311\python.exe -X utf8 attn_ref.py --check
