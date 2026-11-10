# -*- coding: utf-8 -*-
"""查 TG_FP16 开关：ORV 的 GEMM 是否已能整体切到 fp16 输入 + fp32 累加。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
k = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py"), encoding="utf-8").read()
lines = k.splitlines()

for i, ln in enumerate(lines):
    if "TG_FP16" in ln or "_fp16_state" in ln:
        print("%5d: %s" % (i + 1, ln.strip()[:110]))
