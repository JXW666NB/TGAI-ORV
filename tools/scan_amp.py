# -*- coding: utf-8 -*-
"""扫描 ORV 的 AMP 能力：有哪些公开入口、fp16 内核、是否支持 autocast 语义。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "python", "tgai_go", "train")

print("=== amp.py 公开名 ===")
s = io.open(os.path.join(TRAIN, "amp.py"), encoding="utf-8").read()
print([n for n in re.findall(r'^(?:def|class) (\w+)', s, re.M)])
print("amp.py 头部说明：")
print(s[:500])

print()
print("=== dtypes.py 公开名 ===")
d = io.open(os.path.join(TRAIN, "dtypes.py"), encoding="utf-8").read()
print([n for n in re.findall(r'^(?:def|class) (\w+)', d, re.M)])

print()
print("=== cuda.py 里的 fp16/bf16 相关内核 ===")
k = io.open(os.path.join(TRAIN, "cuda.py"), encoding="utf-8").read()
print(sorted(set(re.findall(r'__global__ void (\w*(?:f16|bf16|half)\w*)', k))))

print()
print("=== 是否有 autocast 概念 ===")
for name, src in (("__init__.py", io.open(os.path.join(TRAIN, "__init__.py"), encoding="utf-8").read()),):
    hits = sorted(set(re.findall(r'(?:def|class) (\w*[Aa]utocast\w*)', src)))
    print(name, "autocast 相关:", hits if hits else "无")
