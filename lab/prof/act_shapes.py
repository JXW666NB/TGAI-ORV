# -*- coding: utf-8 -*-
"""前向里"需要梯度"的张量按形状统计 —— 找到激活显存的真实归属。

背景：我原以为 MoE FFN 的 gate/up 是激活大头，但 `situ_glu_orv` 在 CUDA 上
已经走融合内核（`situ_gate_orv`，不留中间量），所以那个判断是错的。
这份探针给 `Tensor._from_gpu` 挂桩，统计**前向期间**创建的
`requires_grad=True` 张量的字节数，按形状聚合再排序 —— 谁大就量化谁。
"""
import argparse
import collections

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()

BY_SHAPE = collections.Counter()
COUNT = collections.Counter()
ON = [False]

_orig = T.Tensor._from_gpu.__func__


def from_gpu_hook(cls, buf, requires_grad=False, _parents=(), **kw):
    if ON[0] and requires_grad:
        try:
            shp = tuple(int(s) for s in buf.shape)
            n = 1
            for s in shp:
                n *= s
            BY_SHAPE[shp] += n * 4          # fp32 字节
            COUNT[shp] += 1
        except Exception:
            pass
    return _orig(cls, buf, requires_grad, _parents, **kw)


T.Tensor._from_gpu = classmethod(from_gpu_hook)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    a = ap.parse_args()

    cfg = dict(d=1024, d_ff=2816, E=4, layers=a.layers, n_q=16, n_kv=2,
               vocab=96000, seq=a.seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=16, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    opt = T.optim.AdamW(params, lr=3e-4)
    csac = cfg["csa_cfg"]
    seq = int(cfg["seq"])
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)

    for _ in range(2):
        ON[0] = False
        BY_SHAPE.clear()
        COUNT.clear()
        ON[0] = True
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        ON[0] = False
        loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
        rt.sync()

    tot = sum(BY_SHAPE.values())
    gi = float(1 << 30)
    ps = rt.pool_stats()
    print("=== 前向创建的 requires_grad 张量（一步）===")
    print("  合计 %.3f GiB / %d 个" % (tot / gi, sum(COUNT.values())))
    print("  对照 pool peak_live = %.3f GiB" % (ps["peak_live_bytes"] / gi))
    print("\n  按形状（前 12，按总字节）：")
    for shp, b in BY_SHAPE.most_common(12):
        print("    %-26s x%-4d %8.2f MB" % (str(shp), COUNT[shp], b / (1 << 20)))


if __name__ == "__main__":
    main()
