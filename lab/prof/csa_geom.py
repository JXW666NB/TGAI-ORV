# -*- coding: utf-8 -*-
"""查 CSA plan 的几何量随 seq 怎么变（特别关注局部窗 lk 是否随 seq 增长）。"""
import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cpu")
print("%8s %8s %8s %8s %6s %10s %10s" % ("seq", "Tlen", "Tc", "lk", "c", "mask_g", "mask_l"))
for seq in (4096, 8192, 16384, 32768):
    cfg = dict(d=128, d_ff=256, E=2, layers=1, n_q=4, n_kv=2, vocab=8192,
               seq=seq, attn_res=False, attn_res_block=2, top_k=2, csa=4,
               n_kv_attn=4, rope_theta=500000.0, csa_compress=4,
               csa_conv_kernel=3, csa_local_window=0)
    CLI.build(cfg, np.random.default_rng(0))
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, seq)
    t = plan._tensors("cpu")
    mg = tuple(int(s) for s in t["mask_g"].shape)
    ml = tuple(int(s) for s in t["mask_l"].shape)
    print("%8d %8d %8d %8d %6d %10s %10s"
          % (seq, plan.Tlen, plan.Tc, plan.lk, plan.c, "x".join(map(str, mg)),
             "x".join(map(str, ml))))
