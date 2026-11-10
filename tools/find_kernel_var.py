# -*- coding: utf-8 -*-
"""找出含 ewise_exp 的内核源码所属变量（打印其前若干行与变量名）。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py")
lines = io.open(p, encoding="utf-8").read().splitlines()

idx = None
for i, ln in enumerate(lines):
    if "ewise_exp(" in ln:
        idx = i
        break
print("ewise_exp 出现在第", idx + 1, "行")

# 向上找赋值语句（形如 NAME = """ 或 NAME = r"""）
for j in range(idx, max(0, idx - 4000), -1):
    ln = lines[j]
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*\s*=\s*r?"""', ln) or re.match(r'^[A-Za-z_][A-Za-z0-9_]*\s*=\s*r?"\\$', ln):
        print("所属常量在第", j + 1, "行:", ln.strip()[:80])
        break
else:
    print("向上 4000 行没找到赋值；打印附近上下文：")
    for j in range(max(0, idx - 12), idx + 3):
        print(j + 1, lines[j][:100])
