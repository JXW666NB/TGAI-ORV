@echo off
cd /d D:\tgai_orv\repro_v21e
echo ==== B=8 fp32 (no-fp16, ??? ORV ???????) ====
C:\Python311\python.exe -X utf8 repro_train.py --data data --steps 30 --batch 8 --seq 2048 --no-fp16 --json out_b8_fp32.json
echo ==== B=2 fp32 (no-fp16) ====
C:\Python311\python.exe -X utf8 repro_train.py --data data --steps 30 --batch 2 --seq 2048 --no-fp16 --json out_b2_fp32.json
echo ==== B=8 fp16 no-ckpt (???????) ====
C:\Python311\python.exe -X utf8 repro_train.py --data data --steps 30 --batch 8 --seq 2048 --no-ckpt --json out_b8_nockpt.json
echo ==== done ====
