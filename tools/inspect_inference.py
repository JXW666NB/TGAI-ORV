# -*- coding: utf-8 -*-
"""查看导出链路与 GO 推理入口，供 skill 文档引用。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def head(rel, n=1):
    p = os.path.join(ROOT, rel)
    if not os.path.isfile(p):
        print("  [缺失] %s" % rel)
        return ""
    s = io.open(p, encoding="utf-8").read()
    print("=== %s ===" % rel)
    for line in s.split("\n")[:n]:
        print("  " + line)
    return s


s = head("python/tgai_go/train/export.py", 1)
i = s.find("def collect")
print("=== collect() ===")
print(s[i:i + 620])

print()
print("=== tgai.py 的 CLI 参数 ===")
t = head("tgai.py", 1)
args = re.findall(r"add_argument\(\s*[\"']([-\w]+)[\"']", t)
print("  ", args)

print()
print("=== tgai.py 的 __main__ / 子命令 ===")
for m in re.finditer(r"(?m)^(?:def |if __name__).*$", t):
    print("  " + m.group(0)[:90])

print()
print("=== docs 里与推理/部署相关的文档 ===")
d = os.path.join(ROOT, "docs")
for fn in sorted(os.listdir(d)):
    if any(k in fn for k in ("GGUF", "TG_V4", "推理", "部署", "引擎", "GO")):
        print("  ", fn)
