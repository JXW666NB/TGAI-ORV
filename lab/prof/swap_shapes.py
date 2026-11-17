# -*- coding: utf-8 -*-
"""print 出 matmul 反向里 `swap_last2` 的实际操作数形状。

上一轮我按"2-D"假设加免转置快路径，结果一次都没命中（内核数不变）。
这里把真实形状记下来，下一步照实测数据写，不再猜。
"""
import argparse
import collections

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()

SHAPES = collections.Counter()
_orig = rt.swap_last2


def swap_hook(buf):
    try:
        SHAPES[tuple(int(s) for s in buf.shape)] += 1
    except Exception:
        pass
    return _orig(buf)


rt.swap_last2 = swap_hook


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
    seq = int(cfg["seq"])
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)

    for _ in range(2):
        SHAPES.clear()
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
        rt.sync()

    print("=== swap_last2 的操作数形状（一步，共 %d 次）===" % sum(SHAPES.values()))
    for shp, n in SHAPES.most_common(12):
        print("  %-28s %4d 次   ndim=%d" % (str(shp), n, len(shp)))


if __name__ == "__main__":
    main()
