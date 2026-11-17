# -*- coding: utf-8 -*-
"""抓出"权重被复制 3~4 份"的元凶。

已知：retention.py 在 4 层 / seq512 上看到 backward 前
  (2816,1024) x120（应 32）、(1024,2816) x60（应 20）、(96000,1024) x6（应 2）。
`_transpose_cuda` 明确"不物化"，只打 `_lazy_T`；所以复制发生在别处。
这里对四条可能路径挂计数：`_materialize_lazy_T`、`rt.transpose2d`、
`rt.swap_last2`、`rt.gemm_nt`。
"""
import argparse
import collections

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)

CNT = collections.Counter()
SHAPES = collections.Counter()
ON = [False]

# 1) Tensor._materialize_lazy_T
_m = T.Tensor._materialize_lazy_T


def mat_hook(self):
    if ON[0]:
        CNT["_materialize_lazy_T"] += 1
        try:
            SHAPES[tuple(int(s) for s in self.shape)] += 1
        except Exception:
            pass
    return _m(self)


T.Tensor._materialize_lazy_T = mat_hook

# 2) 三个 rt 方法（记录**输出**形状）
for nm in ("transpose2d", "swap_last2", "gemm_nt", "gemm_tn"):
    fn = getattr(CLS, nm, None)
    if fn is None:
        continue

    def mk(nm, fn):
        def w(self, *a, **kw):
            out = fn(self, *a, **kw)
            if ON[0]:
                CNT[nm] += 1
                try:
                    SHAPES[(nm, tuple(int(s) for s in out.shape))] += 1
                except Exception:
                    pass
            return out
        return w
    setattr(CLS, nm, mk(nm, fn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    a = ap.parse_args()

    cfg = dict(d=1024, d_ff=2816, E=4, layers=a.layers, n_q=16, n_kv=2,
               vocab=96000, seq=512, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=16, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    opt = T.optim.AdamW(params, lr=3e-4)
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, 512)
    cos, sin = C.rope_tables(csac, 512)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], 512).astype(np.int64)
    tgt = np.roll(ids, -1)

    for _ in range(2):
        ON[0] = False
        CNT.clear()
        SHAPES.clear()
        ON[0] = True
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        ON[0] = False
        loss = T.functional.cross_entropy(logits.reshape(512, cfg["vocab"]), tgt)
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
        rt.sync()

    print("=== 前向期间各落地/转置路径的调用次数 ===")
    for k, v in CNT.most_common():
        print("  %-24s %5d" % (k, v))
    print("\n=== 输出形状 top12 ===")
    for k, v in SHAPES.most_common(12):
        print("  %-40s %5d" % (str(k), v))


if __name__ == "__main__":
    main()
