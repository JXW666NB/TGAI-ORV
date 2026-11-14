# -*- coding: utf-8 -*-
"""停用推理产线（已授权）：禁用相关计划任务，再结束其进程。

可回滚：任务定义已备份在 D:\\_task_backup\\<name>.xml，
恢复用 `schtasks /create /xml <file> /tn <name>`。
"""
import glob
import io
import os
import re
import subprocess

PAT = re.compile(r"(run_tgaigo_server|go_1145\d|1145\d\.bat)", re.I)


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                           encoding="gbk", errors="replace")
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return "ERR %s" % exc


def main():
    tasks = []
    for path in sorted(glob.glob(r"D:\_task_backup\*.xml")):
        try:
            txt = io.open(path, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        if PAT.search(txt):
            tasks.append(os.path.basename(path)[:-4])
    print("命中任务 %d 个：%s" % (len(tasks), ", ".join(tasks)))
    for t in tasks:
        run(["schtasks", "/end", "/tn", t])
        out = run(["schtasks", "/change", "/tn", t, "/disable"])
        print("  disable %-22s %s" % (t, out.strip().splitlines()[0][:40] if out.strip() else ""))

    # 结束仍在跑的服务进程
    killed = 0
    out = run(["wmic", "process", "get", "ProcessId,CommandLine", "/format:csv"])
    for line in out.splitlines():
        if "run_tgaigo_server" not in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        pid = parts[-1] if parts and parts[-1].isdigit() else None
        if pid:
            run(["taskkill", "/f", "/pid", pid])
            print("  killed pid=%s" % pid)
            killed += 1
    print("结束进程 %d 个" % killed)
    print(run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
               "--format=csv,noheader"]).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
