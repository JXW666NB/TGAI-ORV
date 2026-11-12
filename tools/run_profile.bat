@echo off
cd /d D:\tgai_orv\TGAI-ORV-main\tools
echo ########## fp32 ##########
C:\Python311\python.exe -X utf8 profile_block.py
echo ########## fp16 ##########
C:\Python311\python.exe -X utf8 profile_block.py --fp16
