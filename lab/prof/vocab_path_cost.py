# -*- coding: utf-8 -*-
"""vocab 相关的 165 ms 到底在哪：逐个候选量。

候选：梯度缓冲零化 / embedding 前向 / embedding 反向 scatter / CE 反向。
全部按 (S=256, D=1024, V=96000) 的真实规模。
"""
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
S, D, V = 256, 1024, 96000
rng = np.random.default_rng(0)


def timed(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    rt.sync()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    print("=== vocab 相关路径逐个量（S=%d D=%d V=%d）===" % (S, D, V))

    # 1) 分配 + 零化 (V,D) = 393 MB
    def alloc_zero():
        b = rt.empty((V, D), dtype=np.float32)
        rt.zero_(b) if hasattr(rt, "zero_") else b.fill_(0)
    try:
        print("  分配+零化 (V,D):      %8.3f ms" % timed(alloc_zero))
    except Exception as e:
        print("  分配+零化 (V,D):      EXC %r" % (e,))

    # 2) embedding 前向 gather
    W = T.tensor(rng.normal(0, 0.02, (V, D)).astype(np.float32), requires_grad=True)
    ids = rng.integers(0, V, S).astype(np.int64)
    print("  embed 前向 gather:    %8.3f ms"
          % timed(lambda: W[ids] if hasattr(W, "__getitem__") else None))

    # 3) embedding + CE 的完整前向+反向（这条最接近真实路径）
    Wt = T.tensor(rng.normal(0, 0.02, (V, D)).astype(np.float32), requires_grad=True)
    x = T.tensor(rng.normal(0, 1.0, (S, D)).astype(np.float32), requires_grad=True)
    tgt = rng.integers(0, V, S).astype(np.int64)

    def emb_ce():
        logits = x @ Wt.T
        loss = T.functional.cross_entropy(logits, tgt)
        x.grad = None
        Wt.grad = None
        loss.backward()

    print("  head+CE 前向+反向:    %8.3f ms" % timed(emb_ce))


if __name__ == "__main__":
    main()
