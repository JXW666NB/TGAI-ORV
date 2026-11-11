# -*- coding: utf-8 -*-
"""审计 ORV 的梯度检查点实现，定位"激活回收弱于 PyTorch"的原因。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "python", "tgai_go", "train")


def read(p):
    return io.open(p, encoding="utf-8").read() if os.path.isfile(p) else ""


print("=== 1. 哪些模块提到 checkpoint ===")
for base, _, files in os.walk(PKG):
    if "__pycache__" in base:
        continue
    for f in sorted(files):
        if not f.endswith(".py"):
            continue
        p = os.path.join(base, f)
        s = read(p)
        c = len(re.findall(r"(?i)checkpoint", s))
        if c:
            print("  %-28s %d" % (os.path.relpath(p, PKG), c))

print()
print("=== 2. 检查点相关函数/类定义 ===")
for base, _, files in os.walk(PKG):
    if "__pycache__" in base:
        continue
    for f in sorted(files):
        if not f.endswith(".py"):
            continue
        p = os.path.join(base, f)
        s = read(p)
        for m in re.finditer(
                r"(?m)^\s*(?:def|class) (\w*[Cc]heckpoint\w*|\w*[Rr]ecompute\w*)\(", s):
            ln = s[:m.start()].count("\n") + 1
            print("  %-26s:%-5d %s" % (f, ln, m.group(1)))
