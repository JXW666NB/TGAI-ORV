# -*- coding: utf-8 -*-
"""H3 判别：fp16 门控（MIN_OUTPUT / MAX_K）对全尺寸配置梯度的影响。

变体：
  A. fp32 基线（amp 关）
  B. amp 开，默认门控（MIN_OUTPUT=262144, MAX_K=1<<30）  ← 预期复现爆炸
  C. amp 开，MIN_OUTPUT=1e9（fp16 分支彻底关掉）        ← 应回到基线
  D. amp 开，MAX_K=1280（H3：挡住长 K 的 dW GEMM）       ← H3 判据
  E. amp 开，MIN_OUTPUT=262144 但 f16 缓存强制开        ← H1 变体
"""
import numpy as np

import tgai_go.train as T
from tgai_go.train import cuda as CM
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()

CFG = dict(d=1024, d_ff=2816, E=2, layers=2, n_q=16, n_kv=2, vocab=96000, seq=512,
           attn_res=False, attn_res_block=2, top_k=2, csa=4, n_kv_attn=16,
           rope_theta=500000.0, csa_compress=4, csa_conv_kernel=3, csa_local_window=0)


def run(amp, min_out, max_k, label, cache=None):
    CM._FP16_MIN_OUTPUT = min_out
    CM._FP16_MAX_K = max_k
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
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
    if amp:
        T.amp_on(dtype="fp16", weight_cache=True)
        if cache is not None:
            try:
                rt.f16_cache_enable(bool(cache))
            except Exception:
                pass
    logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
    loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
    for p in params:
        p.grad = None
    loss.backward()
    rt.sync()
    lv = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
    tot = 0.0
    for p in params:
        g = getattr(p, "grad", None)
        if g is not None:
            a = np.asarray(g.to_numpy(), np.float64)
            tot += float((a * a).sum())
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
    CM._FP16_MIN_OUTPUT = 262144
    CM._FP16_MAX_K = 1 << 30
    gn = float(np.sqrt(tot))
    print("%-40s loss=%.6f  gnorm=%.6e" % (label, lv, gn))
    return lv, gn


def main():
    print("=== 全尺寸配置（csa=4 / d=1024 / seq=512）===")
    _, gA = run(False, 262144, 1 << 30, "A fp32 基线")
    _, gB = run(True, 262144, 1 << 30, "B amp 默认门控")
    _, gC = run(True, 10 ** 9, 1 << 30, "C amp 但 fp16 全关")
    _, gD = run(True, 262144, 1280, "D amp MAX_K=1280 (H3)")
    _, gE = run(True, 262144, 1 << 30, "E amp + f16 缓存强制开", cache=True)
    print("\n比值（相对 A 基线）:")
    for tag, g in (("B 默认", gB), ("C fp16 全关", gC), ("D MAX_K=1280", gD), ("E 缓存开", gE)):
        print("  %-14s %.2f×" % (tag, g / max(gA, 1e-30)))


if __name__ == "__main__":
    main()
