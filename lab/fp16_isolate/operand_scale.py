# -*- coding: utf-8 -*-
"""量一量：模型里每个 fp16 GEMM 的操作数最大绝对值。

fp16 上限 65504。CSA 的掩码常量是 -1e30 —— 只要它流进任何一次 GEMM 输入，
cast 后就是 -inf，GEMM 输出立刻污染，反向梯度爆炸。
"""
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

STAT = {}
_orig = getattr(CLS, "_sgemm_f16")


def _scan(buf, key):
    """只扫一次每个指针，避免 79 次全量 d2h。"""
    ptr = int(getattr(buf, "ptr", 0))
    if ptr in STAT:
        return STAT[ptr]
    try:
        a = np.asarray(rt.d2h(buf), np.float32)
        m = float(np.abs(a).max()) if a.size else 0.0
    except Exception:
        m = float("nan")
    STAT[ptr] = m
    return m


def _rec(self, transa, transb, m, n, k, A, lda, B, ldb, Cbuf, ldc):
    ma = _scan(A, "A")
    mb = _scan(B, "B")
    key = (int(transa), int(transb), int(m), int(n), int(k))
    prev = STAT.get(("spec", key), (0.0, 0.0))
    STAT[("spec", key)] = (max(prev[0], ma), max(prev[1], mb))
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

    print("=== 每个 GEMM 规格的操作数最大绝对值（fp16 上限 65504）===")
    rows = sorted([k for k in STAT if isinstance(k, tuple) and k and k[0] == "spec"])
    over = []
    for k in rows:
        ma, mb = STAT[k]
        flag = "  <== 溢出 fp16!" if max(ma, mb) > 65504 else ""
        ta, tb, mm, nn, kk = k[1]
        print("  trans=%d,%d m=%4d n=%4d k=%5d   |A|max=%.4g  |B|max=%.4g%s"
              % (ta, tb, mm, nn, kk, ma, mb, flag))
        if max(ma, mb) > 65504:
            over.append(k)
    print("\n溢出 fp16 的规格数:", len(over))


if __name__ == "__main__":
    main()
