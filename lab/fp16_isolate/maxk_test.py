# -*- coding: utf-8 -*-
"""用**正确的模块**改 `_FP16_MAX_K`，验证「长 K 的 fp16 GEMM」是不是元凶。

背景：cuda.py:5974-5975 自己记录过「dW 类形状（K=4992 的长累加、数值对消严重）
会让参数梯度相对误差从 3e-4 恶化到 2e-2」。而本模型里存在 K=96000 的 GEMM
（dlogits(512,96000) @ head(96000,1024)），远超那个量级。

★ 上一轮我用 `from tgai_go.train import cuda as CM` 拿到的是**转发壳**，
  改它的常量无效（实测 MIN_OUTPUT=1e9 时 fp16 仍被调 79 次）。
  这里改用 `sys.modules[type(rt).__module__]` 拿真模块。
"""
import sys

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
MOD = sys.modules[type(rt).__module__]

CFG = dict(d=1024, d_ff=2816, E=2, layers=2, n_q=16, n_kv=2, vocab=96000, seq=512,
           attn_res=False, attn_res_block=2, top_k=2, csa=4, n_kv_attn=16,
           rope_theta=500000.0, csa_compress=4, csa_conv_kernel=3, csa_local_window=0)


def run(max_k, label):
    MOD._FP16_MAX_K = int(max_k)
    MOD._FP16_MIN_OUTPUT = 262144
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
    lv = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
    print("%-30s loss=%.6f  gnorm=%.6e" % (label, lv, float(np.sqrt(tot))))
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
    MOD._FP16_MAX_K = 1 << 30


def main():
    print("=== 限制 fp16 的 K 上限 ===")
    run(1 << 30, "默认 MAX_K=1<<30（应炸）")
    run(1280, "MAX_K=1280（旧护栏）")
    run(8192, "MAX_K=8192")
    run(4096, "MAX_K=4096")


if __name__ == "__main__":
    main()
