# -*- coding: utf-8 -*-
"""找出运行时里 fp16/bf16 GEMM 的入口方法名。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
k = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py"), encoding="utf-8").read()
lines = k.splitlines()

for i, ln in enumerate(lines):
    if "GemmEx" in ln:
        # 往上找最近的 def
        for j in range(i, max(0, i - 40), -1):
            m = re.match(r"\s*def (\w+)\(", lines[j])
            if m:
                print("第 %5d 行  %-28s <- def %s" % (i + 1, ln.strip()[:60], m.group(1)))
                break
