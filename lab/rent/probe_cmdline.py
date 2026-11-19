# -*- coding: utf-8 -*-
"""列出远端 python 进程的 PID + 命令行 + 启动时间（只读）。

跑法：python lab/rent/x.py py D:\orv_bench\probe_cmdline.py
"""
import re
import subprocess


def ps(script):
    p = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                       capture_output=True, text=True, errors="replace")
    return (p.stdout or "") + (p.stderr or "")


out = ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Select-Object ProcessId,CreationDate,CommandLine | Format-List")
# Format-List 每块以空行分隔
for block in re.split(r"\r?\n\r?\n", out):
    if "ProcessId" in block:
        print(block.strip())
        print("-" * 100)

print("== parent chain ==")
print(ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Select-Object ProcessId,ParentProcessId | Format-Table -AutoSize"))
