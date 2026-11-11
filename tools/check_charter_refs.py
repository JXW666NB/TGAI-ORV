# -*- coding: utf-8 -*-
"""校验 PLUGIN_CHARTER.md 里引用的测试与文件都存在。"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = io.open(os.path.join(ROOT, "docs", "PLUGIN_CHARTER.md"), encoding="utf-8").read()

bad = []
refs = re.findall(r"`(tests/[\w./\-]+(?:::\w+)?)`", s)
for ref in refs:
    path, _, func = ref.partition("::")     # 文档用双冒号：tests/x.py::test_y
    p = os.path.join(ROOT, path)
    if not os.path.isfile(p):
        bad.append("%s（文件不存在）" % ref)
        continue
    if func:
        body = io.open(p, encoding="utf-8").read()
        if ("def %s(" % func) not in body:
            bad.append("%s（函数不存在）" % ref)

# docs/ 引用
for ref in re.findall(r"`(docs/[\w.\-]+)`", s):
    if not os.path.isfile(os.path.join(ROOT, ref)):
        bad.append("%s（文档不存在）" % ref)

print("引用的测试/文档数:", len(refs))
print("缺失:", bad or "无")
sys.exit(1 if bad else 0)
