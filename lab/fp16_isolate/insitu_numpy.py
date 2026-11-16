# -*- coding: utf-8 -*-
"""现场 A/B（纯 numpy 参考）：每次 fp16 GEMM 用 numpy fp64 重算比较。

不推导形状 —— 直接用现场缓冲的 `.shape`：
    opA=N ⇒ 缓冲行主序为 (k, lda)，逻辑 op(A) = A[:m, :].T   → (m,k)
    opA=T ⇒ 缓冲行主序为 (m, lda)，逻辑 op(A) = A[:k, :]      → (k,m)
    opB=N ⇒ (k, ldb)，op(B) = B[:k, :]                        → (k,n)
    opB=T ⇒ (n, ldb)，op(B) = B[:n, :].T                      → (k,n)
    C 缓冲行主序 (n, ldc)，逻辑 C = C[:n, :].T                → (ldc,n) → 取 [:m,:].T
"""
import sys

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)

CFG = dict(d=1024, d_ff=2816, E=2, layers=2, n_q=16, n_kv=2, vocab=96000, seq=512,
           attn_res=False, attn_res_block=2, top_k=2, csa=4, n_kv_attn=16,
           rope_theta=500000.0, csa_compress=4, csa_conv_kernel=3, csa_local_window=0)

ORIG = getattr(CLS, "_sgemm_f16")
REPORT = []
N = [0]


def _np_of(buf):
    a = np.asarray(rt.d2h(buf), np.float32)
    return a


def _patched(self, transa, transb, m, n, k, A, lda, B, ldb, Cbuf, ldc):
    ORIG(self, transa, transb, m, n, k, A, lda, B, ldb, Cbuf, ldc)
    N[0] += 1
    if N[0] > 120:                      # 只查前 120 次（够覆盖 fwd+bwd 的一层）
        return
    try:
        rt.sync()
        Aa, Bb, Cc = _np_of(A), _np_of(B), _np_of(Cbuf)
        opA = Aa[:m, :].T if transa == 0 else Aa[:k, :]
        opB = Bb[:k, :] if transb == 0 else Bb[:n, :].T
        ref = opA.astype(np.float64) @ opB.astype(np.float64)        # (m,n)
        got = Cc[:n, :].T[:m, :]
        d = float(np.abs(got - ref).max())
        s = max(float(np.abs(ref).max()), 1e-30)
        REPORT.append((int(transa), int(transb), int(m), int(n), int(k),
                       int(lda), int(ldb), int(ldc), d / s, s,
                       tuple(Aa.shape), tuple(Bb.shape)))
    except Exception as e:
        REPORT.append((int(transa), int(transb), int(m), int(n), int(k),
                       int(lda), int(ldb), int(ldc), float("nan"), 0.0, str(e)[:40], ""))


setattr(CLS, "_sgemm_f16", _patched)


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

    print("总 fp16 调用=%d，已对拍=%d" % (N[0], len(REPORT)))
    ok = [r for r in REPORT if r[8] == r[8] and r[8] <= 2e-3]
    bad = [r for r in REPORT if r[8] != r[8] or r[8] > 2e-3]
    print("正常=%d  异常=%d" % (len(ok), len(bad)))
    for r in bad[:10]:
        print("  BAD t=%d,%d m=%d n=%d k=%d lda=%d ldb=%d ldc=%d rel=%.3e |C|max=%.3g Ashape=%s Bshape=%s %s"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10],
                 r[11] if len(r) > 11 else "", r[12] if len(r) > 12 else ""))
    if not bad:
        worst = max(ok, key=lambda r: r[8])
        print("  最差正常项: t=%d,%d m=%d n=%d k=%d rel=%.3e"
              % (worst[0], worst[1], worst[2], worst[3], worst[4], worst[8]))


if __name__ == "__main__":
    main()
