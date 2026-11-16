# -*- coding: utf-8 -*-
"""三路对照：分清 fp16 的慢是「GEMM 本身」还是「影子/状态」。

1) amp=off
2) amp=gemm（全开）
3) amp=gemm 但把 fp16 门控关掉（_FP16_MIN_OUTPUT=1e9）——
   影子/状态照旧建立，但一次 fp16 GEMM 都不走。
若 (3) ≈ (2) ⇒ 锅在影子/状态；若 (3) ≈ (1) ⇒ 锅在 fp16 GEMM。
"""
import io
import sys
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
MOD = sys.modules[type(rt).__module__]

CFG = dict(d=1024, d_ff=2816, E=4, layers=20, n_q=16, n_kv=2, vocab=96000, seq=512,
           attn_res=False, attn_res_block=2, top_k=2, csa=4, n_kv_attn=16,
           rope_theta=500000.0, csa_compress=4, csa_conv_kernel=3, csa_local_window=0)


def run(mode, label, steps=5, min_out=262144):
    MOD._FP16_MIN_OUTPUT = int(min_out)
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
    rng = np.random.default_rng(0)
    cfg = dict(CFG)
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

    if mode != "off":
        T.amp_on(dtype="fp16", weight_cache=True)
        ptrs = [int(getattr(p, "_gpu", None).ptr) for p in params
                if getattr(p, "_gpu", None) is not None]
        if ptrs:
            rt.f16_cache_pin(ptrs)
        if mode == "nofp16":
            MOD._FP16_MIN_OUTPUT = 10 ** 9

    def step():
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
        for p in params:
            p.grad = None
        loss.backward()
        T.clip_grad_norm_(params, 1.0)
        opt.step()
        rt.sync()

    step()
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t0)
    st = float(np.median(ts[1:]))
    peak = rt.pool_stats()["peak_live_bytes"] / 2**30
    print("%-22s step=%.3f s（中位）  tok/s=%.0f  peak=%.2f GiB"
          % (label, st, seq / st, peak))
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
    MOD._FP16_MIN_OUTPUT = 262144
    return st


def main():
    a = run("off", "1) amp=off")
    b = run("gemm", "2) amp=gemm（门槛 262144）")
    c = run("nofp16", "3) amp 但 fp16 门控关")
    d = run("gemm", "4) amp 门槛 1e6", min_out=10**6)
    e = run("gemm", "5) amp 门槛 4e6", min_out=4*10**6)
    print("\n(2)/(1) = %.2f   (3)/(1) = %.2f   (4)/(1)=%.2f  (5)/(1)=%.2f"
          % (b / a, c / a, d / a, e / a))


if __name__ == "__main__":
    main()
