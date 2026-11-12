@echo off
cd /d D:\tgai_orv\TGAI-ORV-main\tools
echo ########## ORV ##########
C:\Python311\python.exe -X utf8 mem_breakdown.py --impl orv
echo ########## torch ##########
C:\Python311\python.exe -X utf8 mem_breakdown.py --impl torch
