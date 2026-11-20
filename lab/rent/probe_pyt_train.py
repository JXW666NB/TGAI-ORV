# -*- coding: utf-8 -*-
"""摸清 PyTorch 训练脚本的口径：CLI 参数、模型配置、优化器、步数与数据。"""
import io
import re

P = r"D:\orv_bench\v3_pyt_train.py"
s = io.open(P, encoding="utf-8", errors="replace").read()
lines = s.splitlines()

print("=== add_argument ===")
for i, l in enumerate(lines):
    if "add_argument" in l:
        print("  L%d: %s" % (i + 1, l.strip()[:120]))

print("\n=== 关键构造/调用 ===")
pat = re.compile(r"(TGAIV3Config|AdamW|enable_gradient_checkpointing|cross_entropy|"
                 r"seg_id|autocast|step_s|for step in|range\()", re.I)
seen = 0
for i, l in enumerate(lines):
    if pat.search(l) and not l.strip().startswith("#"):
        print("  L%d: %s" % (i + 1, l.strip()[:120]))
        seen += 1
        if seen >= 26:
            break
