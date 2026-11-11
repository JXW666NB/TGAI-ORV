# -*- coding: utf-8 -*-
"""找出 ewise_abs 的完整接线链（内核 → rt 方法 → Tensor 方法 → 反向），
照着加 cos/sin。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CU = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py"),
             encoding="utf-8").read()
INIT = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "__init__.py"),
               encoding="utf-8").read()

print("=== cuda.py: 方法名含 abs ===")
for m in re.finditer(r"(?m)^    def (\w*)abs(\w*)\(.*$", CU):
    print("  ", m.group(0).strip()[:100])

print("=== cuda.py: launch 里用 ewise_abs 的位置 ===")
for m in re.finditer(r"ewise_abs", CU):
    ln = CU[:m.start()].count("\n") + 1
    line = CU[CU.rfind("\n", 0, m.start()) + 1: CU.find("\n", m.start())]
    print("  %-6d %s" % (ln, line.strip()[:100]))

print("=== __init__.py: 含 abs 的定义 ===")
for m in re.finditer(r"(?m)^\s*def (\w*abs\w*)\(.*$", INIT):
    print("  ", INIT[:m.start()].count("\n") + 1, ":", m.group(0).strip()[:100])
