# -*- coding: utf-8 -*-
"""审计：ORV 的 CUDA 后端对齐程度（哪些算子有 CUDA 分支、哪些只有 CPU）。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "__init__.py"),
            encoding="utf-8").read()
k = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py"),
            encoding="utf-8").read()

# Tensor 的公开方法
methods = re.findall(r"(?m)^    def (\w+)\(self", s)
methods = [m for m in methods if not m.startswith("_")]
cuda_branch = set()
for m in re.finditer(r"(?m)^    def (\w+)\(self", s):
    name = m.group(1)
    nxt = re.search(r"(?m)^    def ", s[m.end():])
    body = s[m.end():m.end() + (nxt.start() if nxt else 800)]
    if 'device == "cuda"' in body:
        cuda_branch.add(name)

print("Tensor 公开方法总数:", len(methods))
print("其中显式带 CUDA 分支的:", len(cuda_branch))
missing = [m for m in methods if m not in cuda_branch]
print("没有显式 CUDA 分支的 (%d):" % len(missing))
for i in range(0, len(missing), 6):
    print("   ", ", ".join(missing[i:i + 6]))

print()
print("CUDA 内核数:", k.count("__global__ void"))
print("_cpu_only 调用点:", s.count("_cpu_only("))

# GPU 桥接模块
for f in ("ops_cuda.py", "ops_vk.py", "ops_native.py"):
    p = os.path.join(ROOT, "python", "tgai_go", "train", f)
    if os.path.exists(p):
        t = io.open(p, encoding="utf-8").read()
        print("%-14s 行数 %5d  注册算子 %d" % (f, t.count("\n"),
                                              len(re.findall(r"@register|def (\w+)", t))))
