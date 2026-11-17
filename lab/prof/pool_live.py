# -*- coding: utf-8 -*-
"""驻留显存（权威口径）：读内存池自己的账本，看 backward 前谁在占。

前两版探针都不可信（见 docs/ACTIVATION_MEMORY_INVESTIGATION.md）：
给 `GpuBuf.__init__/__del__` 挂钩子会留下僵尸条目，导致自相矛盾的结果。
这里改用 `rt.pool_dump_live()` —— 池在借出/归还时记账，与 Python 对象生命周期无关。
"""
import argparse

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()


def show(tag, top=14):
    d = rt.pool_dump_live(top=top)
    gi = float(1 << 30)
    print("  [%s] 借出 %d 块 / %.3f GiB" % (tag, d["blocks"], d["bytes"] / gi))
    for sz, n, tot in d["top"]:
        print("      %8d 字节 (n=%-8d) x%-4d %9.2f MB"
              % (sz, sz // 4, n, tot / (1 << 20)))
    return d["bytes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--opt", default="adamw")
    ap.add_argument("--d_ff", type=int, default=2816)
    ap.add_argument("--no-opt", action="store_true",
                    help="不建优化器（只测权重+激活，避免 m/v 挤爆显存）")
    a = ap.parse_args()

    cfg = dict(d=1024, d_ff=a.d_ff, E=4, layers=a.layers, n_q=16, n_kv=2,
               vocab=96000, seq=a.seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=16, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    opt = None if a.no_opt else T.optim.AdamW(params, lr=3e-4)
    csac = cfg["csa_cfg"]
    seq = int(cfg["seq"])
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)
    nparam = sum(int(p._gpu.n) for p in params)
    print("=== %d 层 / seq=%d / %d 参数（fp32 权重 %.3f GiB）==="
          % (a.layers, seq, nparam, nparam * 4 / float(1 << 30)))
    print("\n  ---- build 之后、任何 forward 之前 ----")
    show("after build")

    for step in range(2):
        for p in params:
            p.grad = None
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
        rt.sync()
        if step == 0:
            print("\n  ---- backward 之前 ----")
            show("after forward")
        loss.backward()
        rt.sync()
        if step == 0:
            print("  ---- backward 之后 ----")
            show("after backward")
        if opt is not None:
            opt.step()
        rt.sync()
    ps = rt.pool_stats()
    print("\n  pool: peak_live=%.3f GiB  cached=%.3f GiB"
          % (ps["peak_live_bytes"] / float(1 << 30),
             ps["cached_bytes"] / float(1 << 30)))


if __name__ == "__main__":
    main()
