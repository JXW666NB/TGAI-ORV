# -*- coding: utf-8 -*-
"""校验 DESIGN.md 里引用的文档都存在（防止设计文档指向空气）。"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = io.open(os.path.join(ROOT, "docs", "DESIGN.md"), encoding="utf-8").read()
missing = []
for m in re.finditer(r"`(docs/[\w.\-]+)`", s):
    p = os.path.join(ROOT, m.group(1))
    if not os.path.isfile(p):
        missing.append(m.group(1))
print("引用的文档数:", len(set(re.findall(r"`(docs/[\w.\-]+)`", s))))
print("缺失:", missing or "无")
tags = {t: s.count(t) for t in ("【已实测】", "【已实现】", "【未实现】", "【设计】", "【未验证】")}
print("状态标记:", tags)
sys.exit(1 if missing else 0)
