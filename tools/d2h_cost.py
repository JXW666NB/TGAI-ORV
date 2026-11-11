# -*- coding: utf-8 -*-
"""量化 MoE 每层那次 `rt.d2h(off_buf)` 主机同步的代价。

做法：把 `rt.d2h` 换成一个**返回缓存值、不触发同步**的版本，重测单层 MoE 前向。
两者差值就是这个同步的真实代价（包括它打断流水线的部分，不只是 API 开销）。

运行（需独占 GPU）：python d2h_cost.py
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
        if p.data is not None:
            p.data = rng.normal(0, 0.02, p.shape).astype(np.float32)
    x0 = rng.normal(0, 1, (SEQ, D)).astype(np.float32)

    # ---------- 基线 ----------
    t_base = ev(lambda: moe(T.tensor(x0, requires_grad=True)))
    log("① 基线（真实 d2h 同步）        %7.3f ms/层" % t_base)

    # ---------- 缓存 off，去掉同步 ----------
    real_d2h = rt.d2h
    cache = {}

    def fake_d2h(buf, *a, **kw):
        key = int(getattr(buf, "n", 0))
        if key not in cache:
            cache[key] = real_d2h(buf, *a, **kw)
        return cache[key]

    rt.d2h = fake_d2h
    try:
        # 先跑一次填缓存
        moe(T.tensor(x0, requires_grad=True))
        rt.sync()
        t_nosync = ev(lambda: moe(T.tensor(x0, requires_grad=True)))
        log("② 去掉 d2h 同步（缓存 off）    %7.3f ms/层" % t_nosync)
        saved = t_base - t_nosync
        log("")
        log("同步代价 = %.3f ms/层 ⇒ 28 层约 %.1f ms（占整步 %.0f ms 的 %.1f%%）"
            % (saved, saved * 28, 787.0, 100.0 * saved * 28 / 787.0))
        if saved <= 0:
            log("⇒ 该同步**不是**瓶颈（差值 ≤0，说明被其他开销掩盖或测量噪声）")
    finally:
        rt.d2h = real_d2h


if __name__ == "__main__":
    main()
