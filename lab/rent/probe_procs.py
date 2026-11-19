# -*- coding: utf-8 -*-
"""远端进程/GPU 占用快照（只读）。用于确认基准跑是否真的在跑、有没有残留进程。

跑法：python lab/rent/x.py py D:\orv_bench\probe_procs.py
"""
import subprocess


def run(args):
    p = subprocess.run(args, capture_output=True, text=True, errors="replace")
    return (p.stdout or "") + (p.stderr or "")


print("== tasklist: python ==")
for line in run(["tasklist", "/fi", "IMAGENAME eq python.exe"]).splitlines():
    if "python" in line.lower():
        print("  " + line.strip())

print("\n== nvidia-smi compute apps ==")
print(run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
           "--format=csv"]))

print("== nvidia-smi util ==")
print(run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,"
                          "power.draw,temperature.gpu", "--format=csv"]))

print("== recently written files in D:\\orv_bench ==")
import os
import time
rows = []
for n in os.listdir(r"D:\orv_bench"):
    p = os.path.join(r"D:\orv_bench", n)
    if os.path.isfile(p):
        st = os.stat(p)
        rows.append((st.st_mtime, n, st.st_size))
rows.sort(reverse=True)
for mt, n, sz in rows[:12]:
    print("  %s  %9d  %s" % (time.strftime("%H:%M:%S", time.localtime(mt)), sz, n))
