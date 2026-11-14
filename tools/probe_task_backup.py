# -*- coding: utf-8 -*-
"""备份非系统计划任务的 XML，并统计实验目录占用（只写 D:\\_task_backup）。"""
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


def dir_size(path):
    total = 0
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
                n += 1
            except Exception:
                pass
    return n, total


def main():
    out = run(["schtasks", "/query", "/fo", "csv", "/v"])
    rows = list(csv.reader(io.StringIO(out)))
    hdr = [h.strip().lower() for h in rows[0]]
    i_name = hdr.index("taskname")
    i_run = hdr.index("task to run") if "task to run" in hdr else -1
    bk = r"D:\_task_backup"
    os.makedirs(bk, exist_ok=True)
    keep = []
    for r in rows[1:]:
        if len(r) <= i_name:
            continue
        name = r[i_name].strip()
        if not name or "microsoft" in name.lower():
            continue
        runto = r[i_run].strip() if 0 <= i_run < len(r) else ""
        keep.append((name, runto))
        xml = run(["schtasks", "/query", "/tn", name, "/xml"])
        safe = name.replace("\\", "_").strip("_")
        try:
            io.open(os.path.join(bk, safe + ".xml"), "w", encoding="utf-8",
                    errors="replace").write(xml)
        except Exception:
            pass
    with io.open(os.path.join(bk, "tasks.csv"), "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "run"])
        w.writerows(keep)
    print("已备份 %d 个任务到 %s" % (len(keep), bk))
    for p in (r"D:\tayauotmuc", r"D:\tgai", r"D:\orv_pip_0962", r"D:\orv_src"):
        if os.path.isdir(p):
            n, sz = dir_size(p)
            print("%-18s %6d 文件  %8.1f MB" % (p, n, sz / 2**20))
    print("\n--- 按前缀统计 ---")
    from collections import Counter
    c = Counter()
    for name, runto in keep:
        key = name.split("-")[0].strip("\\") if "-" in name else name.strip("\\")
        c[key] += 1
    for k, v in c.most_common():
        print("%-14s %d" % (k, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
