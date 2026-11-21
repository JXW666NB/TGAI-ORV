#!/bin/bash
# 读 gn3_<tag>_r<rank>.log，抽出 gnorm（兼容 %9.3f 的前导空格），并逐位比较两个 rank。
PY=/root/miniconda3/bin/python
$PY - <<'PY'
import re

def gn(path):
    out = []
    try:
        for line in open(path, encoding="utf-8", errors="replace"):
            m = re.search(r"gnorm=\s*([0-9.]+)", line)
            if m:
                out.append(m.group(1))
    except OSError:
        pass
    return out

for tag in ("ovOFF", "ovDEF"):
    a = gn("/root/gn3_%s_r0.log" % tag)
    b = gn("/root/gn3_%s_r1.log" % tag)
    print("== %s ==" % tag)
    print("  rank0:", a)
    print("  rank1:", b)
    print("  逐位相同:", a == b if (a and b) else "（没取到 gnorm）")
PY
