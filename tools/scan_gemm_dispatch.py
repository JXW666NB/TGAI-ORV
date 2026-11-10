# -*- coding: utf-8 -*-
"""定位 `_sgemm` 里 fp16 快路径的分派点，为 bf16 接线做准备。"""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py"), encoding="utf-8").read()

for needle in ('_sgemm_f16(', '_sgemm_bf16('):
    idx = 0
    print("===", needle)
    while True:
        i = s.find(needle, idx)
        if i < 0:
            break
        ln = s[:i].count("\n") + 1
        line = s[s.rfind("\n", 0, i) + 1:s.find("\n", i)]
        print("  L%-5d %s" % (ln, line.strip()[:100]))
        idx = i + 1
