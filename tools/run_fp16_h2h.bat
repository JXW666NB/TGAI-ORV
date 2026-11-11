@echo off
set TG_FP16=1
C:\Python311\python.exe -X utf8 D:\tgai_orv\TGAI-ORV-main\tools\h2h_clean.py --impl orv --arch dense --dtype fp32
