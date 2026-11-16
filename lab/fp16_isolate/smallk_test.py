# -*- coding: utf-8 -*-
"""只把「K 很小」的 GEMM 退回 fp32，看梯度是否恢复。

模型里出现了 k=2 / k=128 这种规格的 fp16 GEMM —— 内维极小，
fp16 输入量化没有任何累加摊薄，最坏情况。若块内转 fp32 后梯度恢复正常，
说明问题就在这些小 K 规格上（而不是 fp16 GEMM 整体不可用）。
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
KTH = [0]
HITS = [0]


def _patched(self, transa, transb, m, n, k, A, lda, B, ldb, Cbuf, ldc):
    if int(k) < KTH[0]:
        HITS[0] += 1
        MOD._FP16_MIN_OUTPUT = 10 ** 9      # 临时全关 → 走 fp32
        try:
            self._sgemm(transa, transb, m, n, k, A, lda, B, ldb, Cbuf, ldc)
        finally:
            MOD._FP16_MIN_OUTPUT = 262144
        return
    return ORIG(self, transa, transb, m, n, k, A, lda, B, ldb, Cbuf, ldc)


setattr(CLS, "_sgemm_f16", _patched)


def run(kth, label):
    KTH[0] = kth
    HITS[0] = 0
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
    MOD._FP16_MIN_OUTPUT = 262144
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
    tot = 0.0
    for p in params:
        g = getattr(p, "grad", None)
        if g is not None:
            a = np.asarray(g.to_numpy(), np.float64)
            tot += float((a * a).sum())
    print("%-34s gnorm=%.6e   改走 fp32 的次数=%d"
          % (label, float(np.sqrt(tot)), HITS[0]))
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()


def main():
    print("=== 只把 K < 阈值 的 GEMM 退回 fp32 ===")
    run(0, "全部 fp16（基线，应炸）")
    run(4, "K<4 退回")
    run(64, "K<64 退回")
    run(256, "K<256 退回")


if __name__ == "__main__":
    main()
