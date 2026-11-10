# -*- coding: utf-8 -*-
"""审计 interop（PyT 迁移）的真实覆盖：支持哪些 torch 类型、缺哪些、缺了会怎样。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "interop.py"),
            encoding="utf-8").read()

print("from_torch 支持的 torch 类型:")
for n in sorted(set(re.findall(r'name == "(\w+)"', s))):
    print("   ", n)

i = s.find("raise InteropError")
print()
print("不支持时的行为:")
print(s[max(0, i - 200):i + 340])
