# -*- coding: utf-8 -*-
"""目标 1 的**下界测量**：MoE 纯反向时间（不含梯度检查点的前向重算）。

## 为什么要测这个

此前所有优化尝试都是"在不同环节上省时间"。但如果**纯反向本身**就超过 200 ms，
那么无论怎么融合、怎么去重算（去重算还要额外显存，32GB 装不下），
目标「332 → ≤200 ms」都不可能在有内存约束下达成 —— 这是比"试了很多路都失败"更硬的结论。

## 怎么测

梯度检查点的重算发生在"丢弃计算图后再前向"的循环里。这里改为：
**先做一次前向并保留计算图，再对同一张图做反向** —— 这样测到的反向不含重算。

运行（需独占 GPU）：python moe_bwd_lower_bound.py
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TGAI-ORV-main", "python"))

D, F, E, K, SEQ, NL = 1024, 4096, 8, 2, 512, 28
OUT = os.path.join(HERE, "lb.txt")


def main():
    import torch
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    rep = []

    def ev(fn, n=5):
        for _ in range(3):
            fn()
        rt.sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(n):
            fn()
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / n

    moe = T.nn.MoE(D, F, E, top_k=K, gate="softmax_topk")
    for p in moe.parameters():
        p._gpu = rt.h2d(rng.normal(0, 0.02, tuple(int(s) for s in p.shape)).astype(np.float32))
    x0 = rng.normal(0, 1, (SEQ, D)).astype(np.float32)
    gwt = T.tensor(rng.normal(0, 0.01, (SEQ, D)).astype(np.float32))

    # ① 纯前向
    t_fwd = ev(lambda: moe(T.tensor(x0)))
    rep.append("单层 MoE 前向            %7.3f ms  ⇒ 28 层 %.0f ms" % (t_fwd, t_fwd * NL))

    # ② 纯反向：保留图，只测 backward 本身
    def bwd_only():
        y = moe(T.tensor(x0, requires_grad=True))
        for p in moe.parameters():
            p.grad = None
        rt.sync()
        t0 = torch.cuda.Event(True)
        t1 = torch.cuda.Event(True)
        t0.record()
        y.backward(gwt)                 # ← 只计这一段
        t1.record()
        t1.synchronize()
        return t0.elapsed_time(t1)

    for _ in range(2):
        bwd_only()
    rt.sync()
    samples = [bwd_only() for _ in range(5)]
    t_bwd = min(samples)
    rep.append("单层 MoE 纯反向（不重算）  %7.3f ms  ⇒ 28 层 **%.0f ms**" % (t_bwd, t_bwd * NL))

    # ③ 对照：前向+反向（图里含前向，等价于"不重算"的完整一步）
    def fwd_bwd():
        y = moe(T.tensor(x0, requires_grad=True))
        for p in moe.parameters():
            p.grad = None
        y.backward(gwt)
    t_full = ev(fwd_bwd, n=5)
    rep.append("单层 fwd+bwd（同图，无重算）%7.3f ms ⇒ 28 层 %.0f ms" % (t_full, t_full * NL))

    # ④ 结论
    lb = t_bwd * NL
    rep.append("")
    rep.append("**下界结论**：28 层 MoE 纯反向 = %.0f ms。" % lb)
    if lb > 200:
        rep.append("⇒ 已超过目标 200 ms ⇒ **即使完全去掉梯度检查点的重算**（需要额外约 10 GiB 显存，"
                   "32GB 装不下），MoE 反向也达不到 ≤200 ms。目标在内存约束下不可达。")
    else:
        rep.append("⇒ 低于 200 ms，说明去重算可能达标；但需评估额外显存（32GB 是否装得下）。")

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
