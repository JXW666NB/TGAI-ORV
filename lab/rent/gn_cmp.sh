#!/bin/bash
# 只做比对，不跑训练（dump 文件已在 /root/d_plain.rank0.step0 / d_ov.rank0.step0）
PY=/root/miniconda3/bin/python
$PY - <<'PY'
import glob
import json
import math


def load(prefix):
    files = sorted(glob.glob(prefix + ".rank0.step*"))
    if not files:
        return None, None
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f), files[-1]


a, fa = load("/root/d_plain")
b, fb = load("/root/d_ov")
print("  plain 文件:", fa)
print("  ov    文件:", fb)
if a is None or b is None:
    raise SystemExit("缺 dump")

ra, rb = a["rows"], b["rows"]
print("  参数个数: plain=%d ov=%d" % (len(ra), len(rb)))
sa = sum(r[2] for r in ra if isinstance(r[2], float))
sb = sum(r[2] for r in rb if isinstance(r[2], float))
print("  梯度平方和: plain=%.4f ov=%.4f 比值=%.6f" % (sa, sb, sb / sa))
print("  gnorm:      plain=%.6f ov=%.6f 比值=%.6f"
      % (math.sqrt(sa), math.sqrt(sb), math.sqrt(sb / sa)))

bad = []
for x, y in zip(ra, rb):
    i, n, va, vy = x[0], x[1], x[2], y[2]
    if not isinstance(va, float) or not isinstance(vy, float):
        bad.append((i, n, va, vy, None))
    elif va > 0 and abs(vy / va - 1.0) > 0.02:
        bad.append((i, n, va, vy, vy / va))
print("  偏差 >2%% 的参数: %d / %d" % (len(bad), len(ra)))
tot_bad = sum(r[2] for r in bad if isinstance(r[2], float))
print("  这些参数占 plain 梯度平方和的: %.2f%%" % (100.0 * tot_bad / sa if sa else -1))
for i, n, va, vy, r in bad[:12]:
    print("    #%-4d numel=%-8d plain=%.6f ov=%.6f ratio=%s"
          % (i, n, va, vy, ("%.4f" % r) if r else "?"))
PY
