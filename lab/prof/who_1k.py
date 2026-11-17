# -*- coding: utf-8 -*-
"""追查 `1024x1024`（n=1,048,576）的 750 块是who 分配的。

实测（1.5B / 24 层 / seq1024）：forward 结束时该尺寸有 **750 块 = 3.0 GiB**，
而这类权重（注意力 q/k/v/o 投影）只有 24×4 = 96 个。
多出的 ~650 块（2.6 GiB）如果能消除，收益接近 bf16 量化激活，而且**可能无损**。

手法：给 `rt.empty` 挂桩，只统计该尺寸，按**调用栈里第一个库内非 cuda.py 的帧**归因。
（只记分配事件，不依赖对象生命周期，所以不会有僵尸条目问题。）
"""
import argparse
import collections
import traceback

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)

TARGET_N = 1048576
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--d_ff", type=int, default=3328)
    a = ap.parse_args()

    cfg = dict(d=1024, d_ff=a.d_ff, E=4, layers=a.layers, n_q=16, n_kv=2,
               vocab=96000, seq=a.seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=16, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    csac = cfg["csa_cfg"]
    seq = int(cfg["seq"])
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], seq).astype(np.int64)

    ON[0] = True
    logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
    ON[0] = False
    rt.sync()

    d = rt.pool_dump_live(top=3)
    print("=== %d 层 / seq=%d：forward 后池借出 %d 块 / %.3f GiB ==="
          % (a.layers, seq, d["blocks"], d["bytes"] / float(1 << 30)))
    for sz, cnt, tot in d["top"]:
        print("      n=%-10d x%-4d %9.2f MB" % (sz // 4, cnt, tot / (1 << 20)))
    print("\n=== forward 期间分配 n=%d 的调用点（共 %d 次）==="
          % (TARGET_N, sum(SITES.values())))
    for s, n in SITES.most_common(12):
        print("  %-32s %5d" % (s, n))


if __name__ == "__main__":
    main()
