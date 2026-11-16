# -*- coding: utf-8 -*-
"""录下模型里每一次 fp16 GEMM 的完整参数，找"ld ≠ 紧凑布局"的那些。

独立对拍全过，但模型上下文里可能传入了**非紧凑**操作数
（A.n 不等于 lda*k，或 ldb 不等于规定的维度）—— 那种情况下
`_sgemm_f16` 内部的 `cast(A.ptr, A.n)` 会按错误的元素数搬运，
结果就错了。本脚本把可疑调用全部打出来。
"""
import numpy as np

import tgai_go.train as T
import sys

from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)

CFG = dict(d=1024, d_ff=2816, E=2, layers=2, n_q=16, n_kv=2, vocab=96000, seq=512,
           attn_res=False, attn_res_block=2, top_k=2, csa=4, n_kv_attn=16,
           rope_theta=500000.0, csa_compress=4, csa_conv_kernel=3, csa_local_window=0)

REC = []
_orig = getattr(CLS, "_sgemm_f16")


def _rec(self, transa, transb, m, n, k, A, lda, B, ldb, Cbuf, ldc):
    try:
        an = int(getattr(A, "n", -1))
        bn = int(getattr(B, "n", -1))
    except Exception:
        an = bn = -1
    REC.append((transa, transb, int(m), int(n), int(k), int(lda), int(ldb), int(ldc), an, bn))
    return _orig(self, transa, transb, m, n, k, A, lda, B, ldb, Cbuf, ldc)


setattr(CLS, "_sgemm_f16", _rec)


def main():
    rng = np.random.default_rng(1234)
    cfg = dict(CFG)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    seq = int(cfg["seq"])
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(7).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)
    T.amp_on(dtype="fp16", weight_cache=True)
    logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
    loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
    for p in params:
        p.grad = None
    loss.backward()
    rt.sync()
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()

    print("fp16 GEMM 调用数:", len(REC))
    # 紧凑要求：op=N 时该操作数需要 lda>=m 且元素数 >= lda*k；op=T 时 >= lda*(该维)
    bad = []
    seen = {}
    for (ta, tb, m, n, k, lda, ldb, ldc, an, bn) in REC:
        ka = m if ta == 0 else k            # op(A) 是 m×k
        kb = k if tb == 0 else n            # op(B) 是 k×n
        needA = lda * (ka if ta == 0 else ka)
        needB = ldb * (kb if tb == 0 else kb)
        key = (ta, tb, m, n, k, lda, ldb)
        seen[key] = seen.get(key, 0) + 1
        if an > 0 and an < needA:
            bad.append(("A 太短", key, an, needA))
        if bn > 0 and bn < needB:
            bad.append(("B 太短", key, bn, needB))
    print("\n不同的 (transa,transb,m,n,k,lda,ldb) 组合:", len(seen))
    for key, cnt in sorted(seen.items(), key=lambda x: -x[1])[:12]:
        print("  %-34s ×%d" % (str(key), cnt))
    print("\n可疑调用（缓冲长度不足）:", len(bad))
    for b in bad[:10]:
        print("  ", b)


if __name__ == "__main__":
    main()
