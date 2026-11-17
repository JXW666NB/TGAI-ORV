# -*- coding: utf-8 -*-
"""在 32K 上做一次"驻留归属"：forward 结束时谁在占显存（池账本口径）。"""
import argparse

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=32768)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    a = ap.parse_args()
    seq = a.seq

    cfg = dict(d=a.d, d_ff=256, E=2, layers=a.layers, n_q=4, n_kv=2,
               vocab=8192, seq=seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=4, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)
    nparam = sum(int(p._gpu.n) for p in params)

    print("=== seq=%d L=%d d=%d  参数 %.2f GiB  TG_CSA_ATTN_CQ=%s ==="
          % (seq, a.layers, a.d, nparam * 4 / float(1 << 30), C._ATTN_CQ[0]))
    d0 = rt.pool_dump_live(top=5)
    print("  build 后: %d 块 / %.2f GiB" % (d0["blocks"], d0["bytes"] / float(1 << 30)))
    logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
    rt.sync()
    d1 = rt.pool_dump_live(top=14)
    print("  forward 后: %d 块 / %.2f GiB" % (d1["blocks"], d1["bytes"] / float(1 << 30)))
    for sz, cnt, tot in d1["top"]:
        print("      n=%-10d x%-4d %9.2f MB" % (sz // 4, cnt, tot / (1 << 20)))
    loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
    loss.backward(T.tensor(np.array(1.0, np.float32)))
    rt.sync()
    d2 = rt.pool_dump_live(top=6)
    print("  backward 后: %d 块 / %.2f GiB" % (d2["blocks"], d2["bytes"] / float(1 << 30)))
    for sz, cnt, tot in d2["top"]:
        print("      n=%-10d x%-4d %9.2f MB" % (sz // 4, cnt, tot / (1 << 20)))


if __name__ == "__main__":
    main()
