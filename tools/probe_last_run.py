# -*- coding: utf-8 -*-
"""按「上次运行时间」排序计划任务，找出最近被触发的（只读）。"""
import csv
import io
import subprocess


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                           encoding="gbk", errors="replace")
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return "ERR %s" % exc


def main():
    out = run(["schtasks", "/query", "/fo", "csv", "/v"])
    rows = list(csv.reader(io.StringIO(out)))
    hdr = [h.strip().lower() for h in rows[0]]

    def idx(*names):
        for n in names:
            if n in hdr:
                return hdr.index(n)
        return -1

    i_name = idx("taskname", "任务名") if idx("taskname", "任务名") >= 0 else 0
    i_last = idx("last run time", "上次运行时间")
    i_next = idx("next run time", "下次运行时间")
    i_run = idx("task to run", "要运行的任务")
    i_stat = idx("status", "状态")
    recs = []
    for r in rows[1:]:
        if len(r) <= i_name or not r[i_name].strip():
            continue
        if "microsoft" in r[i_name].lower():
            continue
        last = r[i_last].strip() if 0 <= i_last < len(r) else ""
        recs.append((last, r[i_name].strip(),
                     r[i_stat].strip() if 0 <= i_stat < len(r) else "",
                     r[i_next].strip() if 0 <= i_next < len(r) else "",
                     r[i_run].strip() if 0 <= i_run < len(r) else ""))
    # 按时间字符串排序（YYYY/MM/DD HH:MM:SS 可直接比较）
    recs.sort(key=lambda x: x[0], reverse=True)
    print("最近被触发的 15 个任务：")
    for last, name, stat, nxt, runto in recs[:15]:
        print("%-20s | %-8s | next=%-20s | %s" % (last[:19], stat[:8], nxt[:19],
                                                  name[:26]))
        if runto:
            print("                       -> %s" % runto[:90])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
