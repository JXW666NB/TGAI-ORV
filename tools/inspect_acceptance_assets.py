# -*- coding: utf-8 -*-
"""盘点「CUDA 收尾」阶段已有的验证资产，避免计划里重复造。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def exists(*p):
    return os.path.isfile(os.path.join(ROOT, *p))


print("=== 验收/长跑类脚本 ===")
t = os.path.join(ROOT, "tools")
if os.path.isdir(t):
    for f in sorted(os.listdir(t)):
        if not f.endswith(".py"):
            continue
        if re.search(r"(?i)accept|converg|longrun|long_run|parity|align|train_curve|loss_curve", f):
            print("  ", f, os.path.getsize(os.path.join(t, f)))

print()
print("=== 关键文件是否存在 ===")
for rel in ("tools/accept.py", "tools/pytorch_loss_curve.py",
            "tools/qwen3_tg_parity.py", "examples/beginner_pipeline.py",
            "docs/QUICKSTART_TRAIN.md", "docs/ROADMAP_3B_PERF.md",
            "python/tgai_go/train/dist.py", "docs/DP_2GPU_RUNBOOK.md"):
    print("  %-40s %s" % (rel, "YES" if exists(*rel.split("/")) else "NO"))

print()
print("=== 最长训练步数的证据（在 docs 里找 step 数）===")
d = os.path.join(ROOT, "docs")
pats = [r"(\d{3,5})\s*步", r"steps?[=:]\s*(\d{3,5})"]
hits = {}
for f in sorted(os.listdir(d)):
    if not f.endswith(".md"):
        continue
    s = io.open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
    for p in pats:
        for m in re.finditer(p, s):
            n = int(m.group(1))
            if n >= 200:
                hits[f] = max(hits.get(f, 0), n)
for f, n in sorted(hits.items(), key=lambda x: -x[1])[:8]:
    print("  %-46s >= %d 步" % (f, n))
