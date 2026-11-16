# -*- coding: utf-8 -*-
"""固定开销拆解：一步里 fwd / bwd / opt 各占多少，以及各自发多少内核。

ORV 在 4 层/seq512 上 step=0.264 s，而拟合出的"固定开销"约 193 ms ——
本脚本定位它到底在哪一段。
"""
import argparse
import collections
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)

K = collections.Counter()
_orig = rt.launch


def launch(name, grid, block, args=None, *ar, **kw):
    K[str(name)] += 1
    return _orig(name, grid, block, args, *ar, **kw)


rt.launch = launch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n_q", type=int, default=16)
    ap.add_argument("--d_ff", type=int, default=2816)
    ap.add_argument("--vocab", type=int, default=96000)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--csa", type=int, default=4)
    a = ap.parse_args()

    cfg = dict(d=a.d, d_ff=a.d_ff, E=a.E, layers=a.layers, n_q=a.n_q, n_kv=2,
               vocab=a.vocab, seq=a.seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=a.csa, n_kv_attn=a.n_q, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    opt = T.optim.AdamW(params, lr=3e-4)
    seq = int(cfg["seq"])
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)
    print("参数张量数: %d" % len(params))

    def phases():
        t0 = time.perf_counter()
        K.clear()
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
        rt.sync()
        t1 = time.perf_counter()
        k_fwd = sum(K.values())
        K.clear()
        for p in params:
            p.grad = None
        loss.backward()
        rt.sync()
        t2 = time.perf_counter()
        k_bwd = sum(K.values())
        K.clear()
        opt.step()
        rt.sync()
        t3 = time.perf_counter()
        k_opt = sum(K.values())
        return (t1 - t0, t2 - t1, t3 - t2, k_fwd, k_bwd, k_opt)

    phases()        # 预热
    f, b, o, kf, kb, ko = phases()
    print("\n=== 一步的分段（layers=%d seq=%d）===" % (a.layers, a.seq))
    print("  forward : %7.1f ms   内核 %5d" % (f * 1e3, kf))
    print("  backward: %7.1f ms   内核 %5d" % (b * 1e3, kb))
    print("  opt.step: %7.1f ms   内核 %5d   ← 每张量 %.1f 个"
          % (o * 1e3, ko, ko / max(len(params), 1)))
    print("  合计    : %7.1f ms   内核 %5d" % ((f + b + o) * 1e3, kf + kb + ko))


if __name__ == "__main__":
    main()
