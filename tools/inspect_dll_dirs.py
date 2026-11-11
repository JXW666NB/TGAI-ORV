# -*- coding: utf-8 -*-
"""看清 DLL 目录收集逻辑，为"自动发现 torch/lib 与 nvidia/*/lib"做准备。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py"),
            encoding="utf-8").read()

i = s.find("def _dll_dirs(")
j = s.find("\n    def ", i + 10)
if j < 0:
    j = i + 1800
print("=== _dll_dirs() 定义 ===")
print(s[i:j])

k = s.find("def _linux_lib_dirs(")
j2 = s.find("\n    def ", k + 10)
print("=== _linux_lib_dirs() 定义 ===")
print(s[k:(j2 if j2 > 0 else k + 900)])
