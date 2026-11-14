# -*- coding: utf-8 -*-
"""列出占用 V100 显存的进程、它们的父进程，以及相关的计划任务（ASCII 输出）。"""
import subprocess
import sys


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                           encoding="gbk", errors="replace")
        return p.stdout or ""
    except Exception as exc:
        return "ERR %s" % exc


def main():
    print("=== GPU processes ===")
    print(run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
               "--format=csv,noheader"]).strip())
    print("=== process tree ===")
    for line in run(["wmic", "process", "get", "ProcessId,ParentProcessId,Name",
                     "/format:csv"]).splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 4 and parts[0] and parts[0] != "Node":
            name, ppid, pid = parts[1], parts[2], parts[3]
            if "python" in name.lower():
                print("pid=%-7s ppid=%-7s %s" % (pid, ppid, name))
    print("=== scheduled tasks (TGAI / infer) ===")
    out = run(["schtasks", "/query", "/fo", "csv", "/nh"])
    for line in out.splitlines():
        low = line.lower()
        if "tgai" in low or "infer" in low or "11450" in low:
            print(line.strip()[:120])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
