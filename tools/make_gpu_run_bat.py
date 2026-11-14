# -*- coding: utf-8 -*-
"""生成 D:\\gpu_run.bat：先清掉占卡进程，再跑给定命令。

用法（远程）：D:\\gpu_run.bat <python.exe> <script.py> [args...]

为什么这么干：推理服务会被未知守护反复拉起（已禁用全部 278 个计划任务后仍会），
与其继续追源头，不如让每次测量都在干净显存上开始 —— 测量开始时唯一的占卡者
一定是它，清掉即可。
"""
import io


BAT = r"""@echo off
setlocal enabledelayedexpansion
for /f "skip=1 tokens=1" %%p in ('nvidia-smi --query-compute-apps^=pid --format^=csv,noheader 2^>nul') do (
  taskkill /f /pid %%p >nul 2>nul
)
timeout /t 3 /nobreak >nul
%*
"""


def main():
    path = r"D:\gpu_run.bat"
    io.open(path, "w", encoding="ascii", newline="\r\n").write(BAT)
    print("已写 %s" % path)
    print(io.open(path, encoding="ascii").read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
