# -*- coding: utf-8 -*-
"""找出反复拉起推理服务的守护：搜任务 XML + 扫描进程命令行（只读）。"""
import glob
import io
import os
import subprocess

KEYS = ("tgaigo", "11450", "run_tgaigo", "watchdog", "guardian", "supervisor")


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                           encoding="gbk", errors="replace")
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return "ERR %s" % exc


def main():
    print("### 任务 XML 里命中关键字")
    hits = 0
    for path in glob.glob(r"D:\_task_backup\*.xml"):
        try:
            txt = io.open(path, encoding="utf-8", errors="replace").read().lower()
        except Exception:
            continue
        for k in KEYS:
            if k in txt:
                print("  %-46s <- %s" % (os.path.basename(path)[:46], k))
                hits += 1
                break
    print("  共 %d 个任务命中" % hits)

    print("### 进程里命中关键字")
    out = run(["wmic", "process", "get", "ProcessId,CommandLine", "/format:csv"])
    for line in out.splitlines():
        low = line.lower()
        if any(k in low for k in KEYS):
            print("  " + line.strip()[:150])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
