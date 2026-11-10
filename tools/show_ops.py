# -*- coding: utf-8 -*-
"""打印 abs/clamp/log/relu 的 Tensor 方法实现，看 CUDA 分支缺什么。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "__init__.py"), encoding="utf-8").read()

for name in ("abs", "clamp", "log", "relu"):
    m = re.search(r"(?m)^    def %s\(self[^\n]*\):\n" % name, s)
    if not m:
        print("=== %s: 找不到" % name)
        continue
    start = m.start()
    nxt = re.search(r"(?m)^    def ", s[m.end():])
    end = m.end() + (nxt.start() if nxt else 400)
    body = s[start:end]
    print("=== %s ===" % name)
    print(body[:700])
    print()
