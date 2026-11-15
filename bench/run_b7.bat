@echo off
rem 在多分辨率 STFT 损失上对比 ORV（svs-ops 插件包）与 PyTorch。
rem 单独写成 .bat 是因为远端通道对含 & 与引号的长命令不稳定。
set PYTHONPATH=D:\orv_pip_0962\Lib\site-packages
cd /d D:\orv_src
C:\Python311\python.exe -u -X utf8 bench\b7_svs_stft.py > D:\_b11.txt 2>&1
echo EXITCODE=%ERRORLEVEL% >> D:\_b11.txt
