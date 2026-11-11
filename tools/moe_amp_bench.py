# -*- coding: utf-8 -*-
"""MoE 在 AMP（fp16 GEMM）下快多少 —— 这是唯一还有可能达成 −40% 的方向。

背景：单层 MoE fwd+bwd 约 11 ms，其中约 90% 是 GEMM 且只有 50~55% 峰值利用率。
fp16 GEMM 理论约 2×（V100 tensor core 更高），若 MoE 专家投影能走 fp16，
28 层 332 ms → 约 170 ms 即达标。

同时测数值差异（fp32 vs AMP 的 loss 相对差），确保不是"快了但算错"。

运行（需独占 GPU）：python moe_amp_bench.py
"""
import sys

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

    def ev(fn, rep=8):
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
        if p.data is not None:
            p.data = rng.normal(0, 0.02, p.shape).astype(np.float32)
    x0 = rng.normal(0, 1, (SEQ, D)).astype(np.float32)

    def full():
        xx = T.tensor(x0, requires_grad=True)
        yy = moe(xx)
        yy.backward(T.tensor(np.ones((SEQ, D), np.float32)))
        for p in moe.parameters():
            p.grad = None
        return yy

    t_fp32 = ev(full)
    log("① fp32        %.3f ms/层 ⇒ 28 层约 %.0f ms（反向按 2/3 估）" % (t_fp32, t_fp32 * 28))

    # 记录 fp32 的 loss 参考
    ref = np.asarray(full().to_numpy(), np.float32).copy()

    t_amp = None
    try:
        with T.amp():
            t_amp = ev(full)
        log("② AMP(fp16)   %.3f ms/层 ⇒ 28 层约 %.0f ms" % (t_amp, t_amp * 28))
    except Exception as exc:
        log("② AMP 失败：%s: %s" % (type(exc).__name__, str(exc)[:160]))
        return

    # 数值：AMP 下的输出与 fp32 差多少
    with T.amp():
        got = np.asarray(full().to_numpy(), np.float32).copy()
    rel = float(np.abs(got - ref).max() / max(1e-9, float(np.abs(ref).max())))
    log("③ 数值：AMP vs fp32 输出相对差 %.2e ⇒ %s"
        % (rel, "属 fp16 正常范围" if rel <= 5e-3 else "**偏大，需查**"))
    log("")
    if t_amp and t_fp32:
        log("加速比 = %.2fx；28 层 MoE 反向若同步加速：332 ms → 约 %.0f ms（目标 ≤200）"
            % (t_fp32 / t_amp, 332.0 / (t_fp32 / max(t_amp, 1e-9))))


if __name__ == "__main__":
    main()
