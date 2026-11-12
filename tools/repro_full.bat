@echo off
cd /d D:\tgai_orv\repro_v21e
echo ==== ?????? ====
C:\Python311\python.exe -X utf8 synth_data.py --dir data --tokens 20000000
echo ==== B=8 S=2048 (??????) ====
C:\Python311\python.exe -X utf8 repro_train.py --data data --steps 30 --batch 8 --seq 2048 --json out_b8.json
echo ==== B=2 S=2048 (????????) ====
C:\Python311\python.exe -X utf8 repro_train.py --data data --steps 30 --batch 2 --seq 2048 --json out_b2.json
echo ==== done ====
