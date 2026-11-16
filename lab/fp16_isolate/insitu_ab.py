# -*- coding: utf-8 -*-
"""现场 A/B：每次 fp16 GEMM 之后用 fp32 重算一遍，逐调用对比。

不需要推导形状 —— 直接用现场的操作数、ld、trans 标志。
哪个调用的 fp16 结果与 fp32 结果差得离谱，那个就是 bug 所在。
"""
import sys

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)
MOD = sys.modules[CLS.__module__]

CFG = dict(d=1024, d_ff=2816, E=2, layers=2, n_q=16, n_kv=2, vocab=96000, seq=512,
           attn_res=False, attn_res_block=2, top_k=2, csa=4, n_kv_attn=16,
           rope_theta=500000.0, csa_compress=4, csa_conv_kernel=3, csa_local_window=0)

ORIG = getattr(CLS, "_sgemm_f16")
REPORT = []
CALLS = [0]


def _patched(self, transa, transb, m, n, k, A, lda, B, ldb, Cbuf, ldc):
    CALLS[0] += 1
    idx = CALLS[0]
    # 1) 真跑 fp16
    ORIG(self, transa, transb, m, n, k, A, lda, B, ldb, Cbuf, ldc)
    # 2) 用 fp32 重算到临时缓冲（临时关掉 fp16）
    if idx > 400:          # 只查前 400 次，避免太慢
        return
    try:
        C2 = rt.empty(tuple(Cbuf.shape), dtype=np.float32)
        MOD._FP16_MIN_OUTPUT = 10 ** 9
        try:
            self._sgemm(transa, transb, m, n, k, A, lda, B, ldb, C2, ldc)
        finally:
            MOD._FP16_MIN_OUTPUT = 262144
        rt.sync()
        a16 = np.asarray(rt.d2h(Cbuf), np.float32)
        a32 = np.asarray(rt.d2h(C2), np.float32)
        d = float(np.abs(a16 - a32).max())
        s = max(float(np.abs(a32).max()), 1e-30)
        REPORT.append((idx, int(transa), int(transb), int(m), int(n), int(k),
                       int(lda), int(ldb), int(ldc), d / s, s))
    except Exception as e:
        REPORT.append((idx, int(transa), int(transb), int(m), int(n), int(k),
                       int(lda), int(ldb), int(ldc), float("nan"), 0.0))


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

    print("总共 fp16 GEMM 调用:", CALLS[0], " 已对拍:", len(REPORT))
    bad = [r for r in REPORT if r[9] == r[9] and r[9] > 2e-3]
    print("相对误差 > 2e-3 的调用数:", len(bad))
    # 按规格聚合，看哪些规格出错
    agg = {}
    for r in REPORT:
        key = (r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
        prev = agg.get(key)
        if prev is None or (r[9] == r[9] and r[9] > prev[0]):
            agg[key] = (r[9] if r[9] == r[9] else -1.0, r[10])
    print("\n各规格的最差相对误差（前 12）:")
    for key, (rr, scale) in sorted(agg.items(), key=lambda kv: -(kv[1][0] if kv[1][0] == kv[1][0] else -1))[:12]:
        print("  t=%d,%d m=%6d n=%6d k=%6d lda=%6d ldb=%6d ldc=%6d  rel=%9.3e  |C|max=%.3g"
              % (key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7], rr, scale))


if __name__ == "__main__":
    main()
