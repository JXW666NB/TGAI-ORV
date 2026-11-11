# -*- coding: utf-8 -*-
"""找出可用的归约算子（并行 vs 顺序）。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py"),
            encoding="utf-8").read()

print("=== rt.* 方法名里含 sumsq/dot/norm/reduce 的 ===")
for m in re.finditer(r"(?m)^    def (\w*(?:sumsq|dot|norm|reduce)\w*)\(", s):
    print("  ", m.group(1))

print("=== 内核名里含 sumsq/dot/reduce 的 ===")
for m in re.finditer(r"__global__ void (\w*(?:sumsq|dot|reduce)\w*)\(", s):
    print("  ", m.group(1))

print("=== sumsq_sequential 附近有没有并行版本 ===")
i = s.find("def sumsq_sequential")
print(s[max(0, i - 400):i + 200])
