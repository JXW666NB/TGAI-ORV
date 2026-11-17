# -*- coding: utf-8 -*-
"""抓 `build` 阶段对 n=2,883,584（2816x1024 专家权重）的分配现场。

已实测：`CLI.build` 完成后、任何 forward 之前，池里借出 **4.328 GiB**，
而模型权重只有 1.425 GiB —— 建模型阶段多占 3 倍。
其中 n=2,883,584 有 **180 块**（专家权重只应有 40 块）。
这份脚本给 `rt.empty` 挂桩，**在 build 期间**记录该大小的调用点。
"""
import collections
import traceback

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)

TARGET_N = 2883584
SITES = collections.Counter()
ON = [False]

_o_empty = CLS.empty


def site():
    for fr in reversed(traceback.extract_stack()[:-2]):
        fn = fr.filename.replace("\\", "/")
        if "/tgai_go/train/cuda.py" in fn:
            continue
        if "/tgai_go/train/" in fn:
            return "%s:%d" % (fn.rsplit("/", 1)[-1], fr.lineno)
        return "ext:%s:%d" % (fn.rsplit("/", 1)[-1], fr.lineno)
    return "<cuda-only>"


def empty_hook(self, shape, *a, **kw):
    if ON[0]:
        try:
            n = 1
            for s in shape:
                n *= int(s)
        except Exception:
            n = -1
        if n == TARGET_N:
            SITES[site()] += 1
    return _o_empty(self, shape, *a, **kw)


CLS.empty = empty_hook

cfg = dict(d=1024, d_ff=2816, E=4, layers=4, n_q=16, n_kv=2,
           vocab=96000, seq=512, attn_res=False, attn_res_block=2,
           top_k=2, csa=4, n_kv_attn=16, rope_theta=500000.0,
           csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
rng = np.random.default_rng(0)

ON[0] = True
P = CLI.build(cfg, rng)
ON[0] = False
params = CLI.params_of(P)

print("=== build 期间分配 n=%d 的调用点（共 %d 次）===" % (TARGET_N, sum(SITES.values())))
for s, n in SITES.most_common(12):
    print("  %-32s %5d" % (s, n))
print("\n  params 合计 = %d 个浮点（%.3f GiB）"
      % (sum(int(p._gpu.n) for p in params),
         sum(int(p._gpu.n) for p in params) * 4 / float(1 << 30)))
d = rt.pool_dump_live(top=4)
print("  池借出 %d 块 / %.3f GiB" % (d["blocks"], d["bytes"] / float(1 << 30)))
for sz, cnt, tot in d["top"]:
    print("      n=%-10d x%-4d %9.2f MB" % (sz // 4, cnt, tot / (1 << 20)))
