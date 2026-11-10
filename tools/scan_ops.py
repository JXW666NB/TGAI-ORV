# -*- coding: utf-8 -*-
"""扫描：哪些算子在 CUDA 上还是 CPU-only，以及 cuda.py 里已有哪些内核。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
init_py = os.path.join(ROOT, "python", "tgai_go", "train", "__init__.py")
cuda_py = os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py")

s = io.open(init_py, encoding="utf-8").read()
ops = sorted(set(re.findall(r'_cpu_only\("([^"]+)"\)', s)))
print("CUDA 上 CPU-only 的算子 (%d):" % len(ops))
for o in ops:
    print("   ", o)

k = io.open(cuda_py, encoding="utf-8").read()
kern = sorted(set(re.findall(r"__global__ void (\w+)", k)))
print()
print("cuda.py 已有内核 (%d):" % len(kern))
for i in range(0, len(kern), 6):
    print("   ", ", ".join(kern[i:i + 6]))
