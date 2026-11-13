# -*- coding: utf-8 -*-
"""停掉所有 TGAI 相关计划任务（防止产线被自动拉起）。

用户要求：停产线 + 停自动任务，之后不再恢复，专心 ORV。
列出名字以 TGAI 开头的任务并逐个 `/disable`（若正在运行再 `/end`）。
"""
import subprocess
import sys


def run(args):
    return subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def main():
    out = run(["schtasks", "/query", "/fo", "csv", "/nh"]).stdout or ""
    names = []
    for line in out.splitlines():
        if not line.startswith('"'):
            continue
        name = line.split('","')[0].strip('"')
        if name.lstrip("\\").upper().startswith("TGAI"):
            names.append(name)
    print("找到 %d 个 TGAI 任务" % len(names))
    for n in names:
        d = run(["schtasks", "/change", "/tn", n, "/disable"])
        ok = d.returncode == 0
        e = run(["schtasks", "/end", "/tn", n])
        print("  %-24s disable=%s end_rc=%d" % (n, "OK" if ok else "FAIL", e.returncode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
