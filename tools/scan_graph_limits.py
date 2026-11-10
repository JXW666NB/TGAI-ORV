# -*- coding: utf-8 -*-
"""查多图实现里有没有硬上限（槽位数/图数量），以及 arena 栈的管理方式。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
k = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py"), encoding="utf-8").read()

print("=== 图相关的方法 ===")
for m in re.finditer(r"(?m)^    def (graph\w*|\w*arena\w*)\(self[^)]*\):", k):
    print("   ", m.group(1))

print()
print("=== 可能的数量上限 ===")
for pat in (r"_MAX_GRAPHS\s*=\s*\d+", r"max_graphs\s*=\s*\d+", r"len\(self\._graph\w*\)\s*>=?\s*\d+",
            r"\[\]\s*\*\s*\d+", r"slot\s*[<>]=?\s*\d+"):
    for m in re.finditer(pat, k):
        ln = k[:m.start()].count("\n") + 1
        print("  L%-5d %s" % (ln, m.group(0)))
