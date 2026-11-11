# -*- coding: utf-8 -*-
"""查 fp16 GEMM 到底在哪条路径生效，判断"反向 fp16"能不能低成本补上。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py"),
            encoding="utf-8").read()

print("=== 判定 fp16 是否开启的位置 ===")
for m in re.finditer(r"_fp16_state\(\)", s):
    ln = s[:m.start()].count("\n") + 1
    seg = s[ln and s.rfind("\n", 0, m.start()) + 1: s.find("\n", m.start())]
    print("  %-6d %s" % (ln, seg.strip()[:110]))

print()
print("=== gemm_nt / gemm_tn / sgemm 的 fp16 分支 ===")
for fn in ("gemm_nt", "gemm_tn", "sgemm"):
    i = s.find("    def %s(" % fn)
    if i < 0:
        print("  [无] %s" % fn)
        continue
    j = s.find("\n    def ", i + 10)
    body = s[i:j]
    has = "fp16" in body.lower() or "_fp16" in body
    print("  %-8s 含 fp16 逻辑: %s（%d 行）" % (fn, has, body.count("\n")))

print()
print("=== 反向用的 matmul 走哪个函数 ===")
t = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "__init__.py"),
            encoding="utf-8").read()
for kw in ("gemm_tn", "gemm_nt", "sgemm"):
    print("  __init__.py 里 %-8s 出现 %d 次" % (kw, t.count(kw)))
