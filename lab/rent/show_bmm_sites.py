# -*- coding: utf-8 -*-
"""本地查看 probe_bmm_sites.py 产出的 JSON（按每步 FLOPs 排行）。"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"lab\rent\bmm_sites.json"
d = json.load(open(path, encoding="utf-8"))
steps = d["steps"]
last = str(steps[-1])
per = d["per_step"][last]
print("steps=%s  last-step calls=%d" % (steps, sum(per.values())))
fl = {}
for r in d["rows"]:
    fl[r["entry"] + " | " + r["geom"]] = r["flops_total"] / NSTEP / 1e9
rows = [(k, v, per.get(k, 0)) for k, v in fl.items()]
rows.sort(key=lambda x: -x[1])
print("%-60s %6s %10s %9s" % ("key", "n/step", "GFLOP/step", "ms@9.9"))
for k, v, n in rows:
    print("%-60s %6d %10.2f %9.2f" % (k[:60], n, v, v / 9.9))
print("SUM GFLOP/step %.1f（%d 次）" % (sum(v for _, v, _ in rows),
                                        sum(n for _, _, n in rows)))
print("\n--- 栈（每项首现）---")
for r in d["rows"]:
    k = r["entry"] + " | " + r["geom"]
    if per.get(k, 0) == 0:
        continue
    print("%s  n=%d" % (k, per.get(k, 0)))
    print("    " + " <- ".join(reversed(r["stack"])))
