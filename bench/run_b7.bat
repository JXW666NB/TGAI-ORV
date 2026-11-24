@echo off
rem 多分辨率 STFT 损失：ORV（svs-ops 插件包）vs PyTorch。
rem 两条远端约束：① 命令超过几秒会被通道掐掉 ⇒ 必须后台起 + 增量打印；
rem               ② 含 & 与引号的长命令在 PowerShell 里会被吃掉 ⇒ 固化成 .bat。
set PYTHONPATH=D:\orv_pip_0962\Lib\site-packages
cd /d D:\orv_src
start "" /b C:\Python311\python.exe -u -X utf8 bench\b7_svs_stft.py > D:\_b12.txt 2>&1
echo STARTED
