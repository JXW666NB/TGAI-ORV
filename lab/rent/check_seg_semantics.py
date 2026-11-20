# -*- coding: utf-8 -*-
"""看参考实现的 seg 掩码语义（ASCII 行），并核对 ORV 侧 slot_doc 与 seg 的取值。

用来判定 `_mask_global_seg` 里 `cseg == seg` 是否把两种不同的量拿来比。
"""
import io
import json

P = r"D:\orv_bench\model_v3.py"
lines = io.open(P, encoding="utf-8", errors="replace").read().splitlines()

# 找 cseg / cmask / seg_mask 相关行
hits = []
for i, l in enumerate(lines):
    t = l.strip()
    if not t or t.startswith("#"):
        continue
    if any(k in t for k in ("cseg", "cmask", "valid_chunk", "tok_chunk",
                            "seg_mask", "doc_id")):
        hits.append("%d: %s" % (i + 1, t[:120]))
print("=== reference lines ===")
for h in hits[:28]:
    print(h)

# ORV 侧运行时取值
import numpy as np
from tgai_go.train import v3_csa as C

rng = np.random.default_rng(0)
B, T, Tc = 1, 64, 16
ids = rng.integers(0, 100, (B, T)).astype(np.int32)
# 情形 1：无 seg（bench 数据的情形）
lay0 = C._chunk_layout(np.zeros((B, T), np.int64), Tc)
# 情形 2：有 seg，文档号 5 与 9
seg = np.zeros((B, T), np.int64)
seg[0, T // 2:] = 9
lay1 = C._chunk_layout(seg, Tc)

def stats(name, lay, sg):
    sd = np.asarray(lay["slot_doc"])
    seg_a = np.asarray(sg)
    cseg = sd.reshape(B, Tc)
    eq = (cseg.reshape(B, 1, Tc) == seg_a.reshape(B, T, 1))
    out = {
        "slot_doc_uniq": sorted(set(sd.tolist()))[:6],
        "seg_uniq": sorted(set(seg_a.tolist()))[:6],
        "any_equal": bool(eq.any()),
        "frac_true": round(float(eq.mean()), 6),
        "doc_ord_uniq": sorted(set(np.asarray(lay["doc_ord"]).tolist()))[:6],
    }
    print("%-14s %s" % (name, json.dumps(out)))

stats("no-seg", lay0, np.zeros((B, T), np.int64))
stats("with-seg", lay1, seg)
