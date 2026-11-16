# -*- coding: utf-8 -*-
"""规模二分：逐个放大 vocab / seq / d_ff / layers，找出 AMP 从哪一维开始坏。

每个配置都跑 amp=off 与 amp=on 两次，比 loss 与全局梯度范数。
"""
import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_ops as V
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()

BASE = dict(d=256, d_ff=1024, E=2, layers=2, n_q=4, n_kv=2, vocab=4096, seq=64,
            attn_res=False, attn_res_block=2, top_k=2, csa=0, n_kv_attn=4,
            rope_theta=500000.0, csa_compress=4, csa_conv_kernel=3, csa_local_window=0)


def run(cfg, amp_on):
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
    rng = np.random.default_rng(1234)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    seq = int(cfg["seq"])
    dh = int(cfg["d"]) // int(cfg["n_q"])
    plan = None
    if int(cfg.get("csa", 0) or 0) > 0:
        from tgai_go.train import v3_csa as C
        csac = cfg["csa_cfg"]
        plan = C.make_plan(csac, 1, seq)
        cos, sin = C.rope_tables(csac, seq)
    else:
        cos, sin = V.rope_cos_sin(seq, dh, float(cfg["rope_theta"]))
        cos, sin = T.tensor(cos), T.tensor(sin)
    ids = np.random.default_rng(7).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)
    if amp_on:
        T.amp_on(dtype="fp16", weight_cache=True)
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
    return lv, float(np.sqrt(tot))


def probe(tag, **over):
    cfg = dict(BASE)
    cfg.update(over)
    try:
        l0, g0 = run(cfg, False)
        l1, g1 = run(cfg, True)
        print("%-26s loss off/amp = %.6f / %.6f (Δ%.2e)   gnorm off/amp = %.4e / %.4e  ratio=%.3f"
              % (tag, l0, l1, abs(l0 - l1), g0, g1, g1 / max(g0, 1e-30)))
    except Exception as e:
        print("%-26s EXC %s: %s" % (tag, type(e).__name__, str(e)[:70]))


def main():
    print("=== 规模二分（每行 = 只在 BASE 上改一个维度）===")
    probe("BASE d256/seq64/v4096")
    probe("seq=512", seq=512)
    probe("vocab=96000", vocab=96000)
    probe("d_ff=2816", d_ff=2816)
    probe("layers=8", layers=8)
    probe("seq=512 + vocab=96000", seq=512, vocab=96000)
    probe("全部放大", seq=512, vocab=96000, d_ff=2816, layers=8)
    probe("csa=2", csa=2, layers=4)
    probe("csa=2 + 全部放大", csa=2, seq=512, vocab=96000, d_ff=2816, layers=8)
    probe("csa=4 + d=1024 (2层)", csa=4, d=1024, layers=2, n_q=16,
          d_ff=2816, vocab=96000, seq=512)
    probe("csa=4 + 20层 (d=256)", csa=4, layers=20, n_q=4,
          d_ff=2816, vocab=96000, seq=512)
    probe("csa=4 + n_q=16 d=1024 2层", csa=4, d=1024, layers=2, n_q=16,
          d_ff=1024, vocab=4096, seq=64)


if __name__ == "__main__":
    main()
