# -*- coding: utf-8 -*-
"""单层 MoE 的完整 fwd+bwd 耗时，与原语之和对比 —— 找出 332 ms 里未解释的部分。

背景：前向原语合计约 3.1 ms/层（gate 0.99 + up ~1.0 + down 1.12），28 层约 87 ms；
反向按 2× 估算约 174 ms，但实测 MoE 反向是 332 ms ⇒ 约 158 ms 未解释。
本脚本用真实的 MoE 模块（CUDA 路径）测 fwd、bwd，并单独测反向特有的算子。

运行（需独占 GPU）：python moe_layer_bench.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

D, DF, NE, TOPK, SEQ = 1024, 4096, 8, 2, 512


def log(m):
    print(m, flush=True)


def main():
    import torch
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)

    def ev(fn, rep=10):
        for _ in range(3):
            fn()
        rt.sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(rep):
            fn()
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / rep

    moe = T.nn.MoE(D, DF, NE, top_k=TOPK, gate="softmax_topk")
    for p in moe.parameters():
        p.data = rng.normal(0, 0.02, p.shape).astype(np.float32) if p.data is not None else None
    x0 = rng.normal(0, 1, (SEQ, D)).astype(np.float32)

    # ---------- 完整 fwd ----------
    x = T.tensor(x0, requires_grad=True)
    y = moe(x)
    rt.sync()
    t_fwd = ev(lambda: moe(T.tensor(x0, requires_grad=True)))
    log("单层 MoE 前向（真实模块）      %7.2f ms" % t_fwd)

    # ---------- 完整 fwd + bwd ----------
    def full():
        xx = T.tensor(x0, requires_grad=True)
        yy = moe(xx)
        yy.backward(T.tensor(np.ones((SEQ, D), np.float32)))
        for p in moe.parameters():
            p.grad = None
    t_full = ev(full, rep=5)
    log("单层 MoE 前向+反向（真实模块）  %7.2f ms（反向≈%.2f ms）" % (t_full, t_full - t_fwd))

    log("")
    log("换算 28 层：前向 %.0f ms，fwd+bwd %.0f ms，反向 %.0f ms"
        % (t_fwd * 28, t_full * 28, (t_full - t_fwd) * 28))

    # ---------- 反向特有的算子（按真实尺寸） ----------
    log("")
    log("== 反向特有算子（8 专家 × 128 行）==")
    g = rt.h2d(rng.normal(0, 1, (128, DF)).astype(np.float32))
    h = rt.h2d(rng.normal(0, 1, (128, DF)).astype(np.float32))
    dh = rt.h2d(rng.normal(0, 1, (128, DF)).astype(np.float32))
    dx = rt.h2d(rng.normal(0, 1, (128, D)).astype(np.float32))
    tok_buf_holder = {}

    log("silu 反向 ×8                %7.3f ms"
        % ev(lambda: [rt.silu_bwd(g, dh) if hasattr(rt, "silu_bwd") else rt.mul(g, dh) for _ in range(NE)]))
    log("ewise mul ×8（[128,4096]）   %7.3f ms"
        % ev(lambda: [rt.mul(h, dh) for _ in range(NE)]))
    log("scale ×8                    %7.3f ms"
        % ev(lambda: [rt.scale(h, 1.0) for _ in range(NE)]))


if __name__ == "__main__":
    main()
