# -*- coding: utf-8 -*-
"""量 embedding 反向的 scatter-add：(V,D) 目标 + S 行源。

如果这个内核的 grid 是按"源行数"配的，而目标有 98M 元素，
每个线程就要循环几十万次 —— 那它就是 1 层 backward 里那 165 ms。
"""
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_scatter as VS

T.set_default_device("cuda")
rt = T._cuda()
V, D, S = 96000, 1024, 256
rng = np.random.default_rng(0)


def timed(fn, n=3, warm=1):
    for _ in range(warm):
        fn()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    rt.sync()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    acc = T.tensor(np.zeros((V, D), np.float32))
    ids = rng.integers(0, V, S).astype(np.int64)
    src = T.tensor(rng.normal(0, 1e-3, (S, D)).astype(np.float32))

    print("=== embedding 反向 scatter-add（V=%d D=%d S=%d）===" % (V, D, S))
    fn = getattr(VS, "index_add", None)
    if fn is None:
        print("  v3_scatter.index_add 不存在")
        return
    try:
        ms = timed(lambda: fn(acc, ids, src, dim=0))
        moved = V * D * 4
        print("  index_add → (V,D): %8.3f ms  （若按目标 393 MB 算等效 %.0f GB/s）"
              % (ms, moved / (ms * 1e-3) / 1e9))
    except Exception as e:
        print("  调用失败：%r" % (e,))

    # 对照：直接对 (V,D) 做一次乘加，看"正常"的 393 MB 内核该多久
    one = T.tensor(np.ones((V, D), np.float32))
    try:
        ms2 = timed(lambda: one * 1.0001)
        print("  对照：整块 (V,D) 乘一次: %8.3f ms" % ms2)
    except Exception as e:
        print("  对照失败：%r" % (e,))


if __name__ == "__main__":
    main()
