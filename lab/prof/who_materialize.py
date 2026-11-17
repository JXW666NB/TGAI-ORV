# -*- coding: utf-8 -*-
"""定位 `_materialize_lazy_T` 的调用点：每层 3 次权重转置物化（共 72 块 / 0.94 GiB）。

已实测（1.5B / 24 层 / seq1024）：forward 结束时 `(3328,1024)` 有 432 块，
而这类权重是 360 块 ⇒ 多出 72 块 = 24 层 × 3，形状正是 W3 的转置。
`_matmul_cuda` 对 2-D 已能走 `gemm_nt` 免物化，所以这 3 处是走了别的路径。

本脚本记录每次物化的**调用点**与**形状**，据此定位。
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

SITES = collections.Counter()
SHAPES = collections.Counter()
ON = [False]

_o = T.Tensor._materialize_lazy_T


def site():
    for fr in reversed(traceback.extract_stack()[:-2]):
        fn = fr.filename.replace("\\", "/")
        if "/tgai_go/train/cuda.py" in fn:
            continue
        return "%s:%d" % (fn.rsplit("/", 1)[-1], fr.lineno)
    return "<cuda-only>"


def hook(self):
    if ON[0]:
        s = site()
        SITES[s] += 1
        try:
            SHAPES[(s, tuple(int(x) for x in self.shape))] += 1
        except Exception:
            pass
    return _o(self)


T.Tensor._materialize_lazy_T = hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--seq", type=int, default=1024)
    a = ap.parse_args()

    cfg = dict(d=1024, d_ff=3328, E=4, layers=a.layers, n_q=16, n_kv=2,
               vocab=96000, seq=a.seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=16, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    csac = cfg["csa_cfg"]
    seq = int(cfg["seq"])
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], seq).astype(np.int64)

    ON[0] = True
    CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
    ON[0] = False
    rt.sync()

    print("=== forward 期间 lazy_T 物化次数：%d（%d 层）===" % (sum(SITES.values()), a.layers))
    for s, n in SITES.most_common(10):
        print("  %-32s %5d  (%.2f 次/层)" % (s, n, n / float(a.layers)))
    print("\n=== (调用点, 被转置的形状) top10 ===")
    for k, n in SHAPES.most_common(10):
        print("  %-48s %5d" % (str(k), n))


if __name__ == "__main__":
    main()
