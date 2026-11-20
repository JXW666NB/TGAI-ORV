# -*- coding: utf-8 -*-
"""看 PyTorch 训练脚本的模型构造 / 取数 / 训练循环（L55-100）。"""
import io

P = r"D:\orv_bench\v3_pyt_train.py"
lines = io.open(P, encoding="utf-8", errors="replace").read().splitlines()
for i in range(54, min(101, len(lines))):
    print("%3d| %s" % (i + 1, lines[i]))

print("\n=== TGAIV3Config 的字段 ===")
import sys
sys.path.insert(0, r"D:\orv_bench")
try:
    import model_v3 as M
    import dataclasses
    for f in dataclasses.fields(M.TGAIV3Config):
        tag = "  <%s>" % f.type if getattr(f, "type", None) else ""
        print("  %-24s = %r%s" % (f.name, f.default, tag))
except Exception as e:
    print("  取不到:", e)
