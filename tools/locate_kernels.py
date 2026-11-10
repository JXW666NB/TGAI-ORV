# -*- coding: utf-8 -*-
"""定位内核源码字符串常量 + Runtime 的要素级接口名。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
k = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py"), encoding="utf-8").read()

# 1) ewise_exp 内核在哪个字符串常量里
i = k.find("__global__ void ewise_exp(")
seg = k[max(0, i - 3000):i]
m = list(re.finditer(r"(_[A-Z0-9_]+_KERNELS?)\s*=\s*(\"\"\"|r\"\"\")", seg))
print("ewise_exp 属于常量:", m[-1].group(1) if m else "未定位")
print("该常量起始位置:", (max(0, i - 3000) + m[-1].start()) if m else "?")

# 2) Runtime 的要素级方法（含 ewise 的封装）
print()
print("Runtime 里与要素级相关的 def：")
for mm in re.finditer(r"(?m)^    def (\w+)\(self[^)]*\):", k):
    n = mm.group(1)
    if n in ("add", "sub", "mul", "div", "scale", "add_scalar", "mul_scalar", "neg", "abs",
             "clamp", "relu", "where", "maximum", "minimum", "log", "exp", "sqrt", "pow"):
        print("   ", n)
