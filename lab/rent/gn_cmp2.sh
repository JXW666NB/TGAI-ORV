#!/bin/bash
# 看两份 dump 的前几行到底是什么（值类型/量级），再做稳健比对。
PY=/root/miniconda3/bin/python
$PY - <<'PY'
import glob
import json
import math


def load(prefix):
    f = sorted(glob.glob(prefix + ".rank0.step*"))[-1]
    with open(f, encoding="utf-8") as fh:
        d = json.load(fh)
    return d["rows"], f


ra, fa = load("/root/d_plain")
rb, fb = load("/root/d_ov")
print("plain 前 3 行:", ra[:3])
print("ov    前 3 行:", rb[:3])


def val(x):
    return x if isinstance(x, float) else None


def total(rows):
    return sum(v for v in (val(r[2]) for r in rows) if v is not None)


def count_bad_type(rows):
    return sum(1 for r in rows if val(r[2]) is None)


print("non-float 行数: plain=%d ov=%d" % (count_bad_type(ra), count_bad_type(rb)))
sa, sb = total(ra), total(rb)
print("梯度平方和(只算 float): plain=%.6f ov=%.6f" % (sa, sb))
if sa > 0 and sb > 0:
    print("gnorm 比值: %.6f" % math.sqrt(sb / sa))
    pairs, only_a, only_b = [], 0, 0
    for x, y in zip(ra, rb):
        va, vy = val(x[2]), val(y[2])
        if va is None and vy is None:
            continue
        if va is None or vy is None:
            only_a += 1
            continue
        pairs.append((x[0], x[1], va, vy))
    print("可比值对数=%d（单侧缺失 %d）" % (len(pairs), only_a))
    bad = [(i, n, va, vy, vy / va) for (i, n, va, vy) in pairs
           if va > 0 and abs(vy / va - 1.0) > 0.02]
    tot_bad = sum(b[2] for b in bad)
    print("偏差 >2%% 的参数: %d / %d，占 plain 平方和 %.2f%%"
          % (len(bad), len(pairs), 100.0 * tot_bad / sa))
    for i, n, va, vy, r in bad[:12]:
        print("  #%-4d numel=%-8d plain=%.6f ov=%.6f ratio=%.4f" % (i, n, va, vy, r))
else:
    # 有 err 字符串就贴出来
    print("plain 里的非 float 样例:", [r for r in ra if val(r[2]) is None][:3])
    print("ov    里的非 float 样例:", [r for r in rb if val(r[2]) is None][:3])
PY
