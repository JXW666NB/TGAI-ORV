# -*- coding: utf-8 -*-
"""核实 P100 报告点名的迁移缺口：cos/sin、SDPA 签名、分块 CE。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "python", "tgai_go", "train")


def read(rel):
    p = os.path.join(ROOT, rel)
    return io.open(p, encoding="utf-8").read() if os.path.isfile(p) else ""


s = read(os.path.join("python", "tgai_go", "train", "__init__.py"))

print("=== 1. Tensor / functional 是否有 cos / sin ===")
for kw in ("def cos", "def sin", "cos =", "sin ="):
    print("  %-10s %d" % (kw, s.count(kw)))
print("  functional 里含 cos/sin 的行：")
for m in re.finditer(r"(?m)^.*\b(?:cos|sin)\b.*$", s):
    t = m.group(0).strip()
    if "def " in t or "cos" in t.lower():
        print("   ", t[:100])
        if s.count("\n") and len([x for x in re.finditer(r'(?m)^.*\b(?:cos|sin)\b.*$', s)]) > 12:
            break

print()
print("=== 2. scaled_dot_product_attention 签名 ===")
i = s.find("def scaled_dot_product_attention")
print(s[i:i + 700] if i > 0 else "  [未找到]")

print()
print("=== 3. 分块交叉熵接口 ===")
for name in ("linear_cross_entropy", "chunked_ce", "cross_entropy"):
    j = s.find("def %s" % name)
    if j > 0:
        print("  def %s(...) 在第 %d 行" % (name, s[:j].count("\n") + 1))
        print("   ", s[j:j + 260].replace("\n", "\n    "))
