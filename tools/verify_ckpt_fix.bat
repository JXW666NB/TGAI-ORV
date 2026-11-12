@echo off
cd /d D:\tgai_orv\TGAI-ORV-main
echo ==== ????????? GPU ???====
C:\Python311\python.exe -X utf8 -m pytest tests/test_checkpoint_cuda.py tests/test_checkpoint_grad.py -q --tb=short
echo ==== ????????? ====
C:\Python311\python.exe -X utf8 tools\ckpt_scale.py
