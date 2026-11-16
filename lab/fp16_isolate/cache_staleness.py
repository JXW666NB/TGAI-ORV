# -*- coding: utf-8 -*-
"""决定性验证：fp16 权重/激活影子缓存（按指针缓存）是否造成反向错误。

假设：`_cast_f16_cached` 按**指针**缓存 fp16 拷贝。训练时激活值从内存池复用地址，
      同一地址在一步内会先后装不同张量 ⇒ 反向复用了前向的旧 fp16 拷贝 ⇒ 梯度全错。
      这也解释了为什么"每次 GEMM 后强制同步"无效（不是时序问题，是内容陈旧）。
"""
import sys

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()

CFG = dict(d=1024, d_ff=2816, E=2, layers=2, n_q=16, n_kv=2, vocab=96000, seq=512,
           attn_res=False, attn_res_block=2, top_k=2, csa=4, n_kv_attn=16,
           rope_theta=500000.0, csa_compress=4, csa_conv_kernel=3, csa_local_window=0)


def run(amp, weight_cache, label):
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
        T.amp_on(dtype="fp16", weight_cache=weight_cache)
    st = T.amp_state()
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
    print("%-38s loss=%.6f  gnorm=%.6e  cache_on=%s"
          % (label, lv, float(np.sqrt(tot)), st.get("cache", {}).get("on")))
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()


def main():
    print("=== 影子缓存开/关对比（全尺寸配置）===")
    run(False, True, "A fp32 基线")
    run(True, True, "B amp + 缓存开（当前 bug 路径）")
    run(True, False, "C amp + 缓存关（每次重 cast）")


if __name__ == "__main__":
    main()
