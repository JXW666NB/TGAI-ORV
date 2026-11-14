# -*- coding: utf-8 -*-
"""禁用全部非系统计划任务（已授权；全部可从 D:\\_task_backup 恢复）。

恢复单条：schtasks /create /xml "D:\\_task_backup\\<name>.xml" /tn "<name>"
"""
import csv
import io
import os
import subprocess


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                           encoding="gbk", errors="replace")
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return "ERR %s" % exc


def main():
    path = r"D:\_task_backup\tasks.csv"
    if not os.path.exists(path):
        print("缺 tasks.csv")
        return 1
    names = []
    with io.open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].lower() == "task":
                continue
            names.append(row[0])
    ok = fail = 0
    for n in names:
        out = run(["schtasks", "/change", "/tn", n, "/disable"])
        if "错误" in out or "ERROR" in out.upper() or "无法" in out:
            fail += 1
        else:
            ok += 1
    print("禁用完成：成功 %d，失败 %d（共 %d）" % (ok, fail, len(names)))
    # 结束仍在跑的服务
    out = run(["wmic", "process", "where", "name='python.exe'",
               "get", "ProcessId,CommandLine", "/format:list"])
    cur = {}
    killed = 0
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("CommandLine="):
            cur["cmd"] = line[12:]
        elif line.startswith("ProcessId=") and cur.get("cmd"):
            if "run_tgaigo_server" in cur["cmd"]:
                run(["taskkill", "/f", "/pid", line[10:]])
                killed += 1
                print("  killed pid=%s" % line[10:])
            cur = {}
    print("结束进程 %d 个" % killed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
