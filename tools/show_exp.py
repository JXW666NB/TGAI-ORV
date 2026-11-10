# -*- coding: utf-8 -*-
"""打印 Tensor.exp / Tensor.__truediv__ 的 CUDA 分支实现，作为接内核的模板。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "__init__.py"), encoding="utf-8").read()

m = re.search(r"(?m)^    def exp\(self\):\n", s)
if m:
    nxt = re.search(r"(?m)^    def ", s[m.end():])
    print("=== exp ===")
    print(s[m.start():m.end() + (nxt.start() if nxt else 800)][:1100])
