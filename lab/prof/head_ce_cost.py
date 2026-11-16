# -*- coding: utf-8 -*-
"""固定开销归因：head / embedding / CE 这条与层数无关的路径，各占多少。

拟合已知：ORV 一步 = 193 ms 固定 + 17.9 ms/层。
本脚本把"固定"部分逐项量出来（微基准，多次取平均）。

在 V100 上跑：D:\\orv_pip_0962\\Scripts\\python.exe head_ce_cost.py
"""
import argparse
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()


def timed(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    rt.sync()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=96000)
    a = ap.parse_args()
    S, D, V = a.seq, a.d, a.vocab
    rng = np.random.default_rng(0)

    x = rt.h2d(rng.normal(0, 0.02, (S, D)).astype(np.float32))
    Wh = rt.h2d(rng.normal(0, 0.02, (V, D)).astype(np.float32))
    logits = rng.normal(0, 1.0, (S, V)).astype(np.float32)
    labels = rng.integers(0, V, S).astype(np.int64)

    res = {}
    # 1) head 前向：logits = x @ Wh^T
    res["head 前向 GEMM"] = timed(lambda: rt.gemm_nt(x, Wh))
    # 2) CE（前向+反向一体）
    lg = T.tensor(logits.copy(), requires_grad=True)
    def ce():
        lo = T.functional.cross_entropy(lg, labels)
        lo.backward()
    res["CE 前向+反向"] = timed(ce, n=5)
    # 3) head 反向 dX = g @ Wh（K=V）
    g = rt.h2d(rng.normal(0, 1e-3, (S, V)).astype(np.float32))
    res["head 反向 dX（K=%d）" % V] = timed(lambda: rt.sgemm(g, Wh), n=5)
    # 4) head 反向 dW = g^T @ x（K=S）
    res["head 反向 dW（K=%d）" % S] = timed(lambda: rt.gemm_tn(g, x), n=5)
    # 5) embedding gather
    ids = rng.integers(0, V, S).astype(np.int64)
    idx = rt.h2d(ids, dtype=np.int32) if "dtype" in str(rt.h2d.__doc__ or "") else rt.h2d(ids)
    Ew = rt.h2d(rng.normal(0, 0.02, (V, D)).astype(np.float32))
    try:
        res["embedding gather"] = timed(lambda: rt.gather_rows(Ew, rt.h2d(ids.astype(np.int64))), n=20)
    except Exception as e:
        res["embedding gather"] = float("nan")
    # 6) embedding 反向：对 (V,D) 做 scatter-add
    src = rt.h2d(rng.normal(0, 1e-3, (S, D)).astype(np.float32))
    try:
        res["embedding scatter（(V,D) 累加）"] = timed(
            lambda: rt.scatter_add_rows(rt.empty((V, D), dtype=np.float32),
                                        rt.h2d(ids.astype(np.int64)), src), n=5)
    except Exception as e:
        res["embedding scatter（(V,D) 累加）"] = float("nan")

    total = 0.0
    print("=== 固定路径逐项成本（seq=%d d=%d vocab=%d）===" % (S, D, V))
    for k, ms in res.items():
        flag = "" if ms == ms else "  (接口不匹配，未测)"
        if ms == ms:
            total += ms
        print("  %-34s %8.3f ms%s" % (k, ms, flag))
    print("  %-34s %8.3f ms" % ("合计", total))
    print("\n对照：拟合的固定开销 ≈ 193 ms，PyTorch 固定 ≈ 23 ms")


if __name__ == "__main__":
    main()
