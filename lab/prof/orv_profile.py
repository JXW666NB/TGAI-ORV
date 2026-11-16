# -*- coding: utf-8 -*-
"""ORV 侧链路拆解：一步训练发了多少个 GPU 内核、分别是什么。

不看时间（异步计时会失真），看**内核派发次数与种类** ——
派发次数是"调度开销"最直接的代理指标。

在 V100 上跑：
    D:\\orv_pip_0962\\Scripts\\python.exe orv_profile.py --layers 4
"""
import argparse
import collections
import sys

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)

KERNELS = collections.Counter()
BLAS = collections.Counter()
SHAPES = collections.Counter()


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
    ap.add_argument("--amp", default="off")
    a = ap.parse_args()

    # ---- 插桩：内核名计数 ----
    orig_launch = rt.launch

    def launch(name, grid, block, args=None, *ar, **kw):
        KERNELS[str(name)] += 1
        return orig_launch(name, grid, block, args, *ar, **kw)

    rt.launch = launch

    for nm in ("_sgemm", "_sgemm_f16", "gemm_f16_nt_raw", "sgemm_split16",
               "_sgemm_bf16", "grouped_nt", "grouped_nn", "grouped_tn",
               "_grouped_sgemm"):
        fn = getattr(CLS, nm, None)
        if fn is None:
            continue

        def mk(nm, fn):
            def w(self, *ar, **kw):
                BLAS[nm] += 1
                if nm == "_sgemm" and len(ar) >= 5:
                    SHAPES[(int(ar[2]), int(ar[3]), int(ar[4]))] += 1
                return fn(self, *ar, **kw)
            return w
        setattr(CLS, nm, mk(nm, fn))

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
    if a.amp != "off":
        T.amp_on(dtype="fp16", weight_cache=True)

    def one_step():
        KERNELS.clear(); BLAS.clear(); SHAPES.clear()
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
        rt.sync()
        return float(np.asarray(loss.to_numpy()).reshape(-1)[0])

    one_step()                       # 预热（JIT + 池增长）
    KERNELS.clear(); BLAS.clear()
    one_step()                       # 计数这一步

    total = sum(KERNELS.values())
    print("=== ORV 一步（layers=%d d=%d seq=%d csa=%d amp=%s）==="
          % (a.layers, a.d, a.seq, a.csa, a.amp))
    print("内核派发总次数: %d" % total)
    print("GEMM 相关调用: %s" % dict(sorted(BLAS.items(), key=lambda kv: -kv[1])))
    tot_gemm = sum(SHAPES.values())
    print("\nGEMM 形状分布（共 %d 次，%d 种）:" % (tot_gemm, len(SHAPES)))
    for (m, n, k), c in SHAPES.most_common(14):
        print("  m=%5d n=%6d k=%6d  x%d   (%.1f GFLOP/次)"
              % (m, n, k, c, 2.0 * m * n * k / 1e9))

    print("\n按次数排序（前 20）:")
    for nm, cnt in KERNELS.most_common(20):
        print("  %-34s %6d  (%.1f%%)" % (nm, cnt, cnt * 100.0 / max(total, 1)))


if __name__ == "__main__":
    main()
