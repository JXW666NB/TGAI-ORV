# -*- coding: utf-8 -*-
"""把 head / embedding / CE 这条**与层数无关**的固定路径单独量出来。

背景：目标书写着"193 ms 固定开销 + 17.9 ms/层"，所以 4 层时 ORV(264 ms) 反而慢于
PyTorch(235 ms)。但那 193 ms 里 head/CE 占多少，从来没单独量过。
层数不影响这条路径（只在最前/最后各跑一次），所以它直接决定小模型的竞争力。
"""
import argparse
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
_ONE = None


def bench(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    rt.sync()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        rt.sync()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1000.0        # 中位，毫秒


def main():
    global _ONE
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=96000)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    seq, d, V = a.seq, a.d, a.vocab
    _ONE = T.tensor(np.array(1.0, np.float32))
    print("=== 固定路径分段（seq=%d d=%d vocab=%d）===" % (seq, d, V))

    h = T.tensor(rng.normal(0, 0.02, (seq, d)).astype(np.float32), requires_grad=True)
    W = T.tensor(rng.normal(0, 0.02, (d, V)).astype(np.float32), requires_grad=True)
    tgt = rng.integers(0, V, seq).astype(np.int64)

    t_head_f = bench(lambda: h @ W)

    def fwd_bwd_head():
        h.grad = None
        W.grad = None
        y = h @ W
        loss = T.functional.cross_entropy(y, tgt)
        loss.backward(_ONE)

    t_head_fb = bench(fwd_bwd_head, n=3)

    logits = T.tensor(rng.normal(0, 1.0, (seq, V)).astype(np.float32), requires_grad=True)
    t_ce_f = bench(lambda: T.functional.cross_entropy(logits, tgt))

    def fwd_bwd_ce():
        logits.grad = None
        loss = T.functional.cross_entropy(logits, tgt)
        loss.backward(_ONE)

    t_ce_fb = bench(fwd_bwd_ce, n=3)

    E = T.tensor(rng.normal(0, 0.02, (V, d)).astype(np.float32), requires_grad=True)
    ids = rng.integers(0, V, seq).astype(np.int64)
    t_emb_f = bench(lambda: E[ids])

    def emb_fb():
        E.grad = None
        y = E[ids]
        y.backward(_ONE)
        rt.sync()

    t_emb_fb = bench(emb_fb, n=3)

    print("\n  项目                      前向       前向+反向")
    print("  head GEMM (%dx%dx%d)   %7.2f ms %8.2f ms"
          % (seq, d, V, t_head_f, t_head_fb))
    print("  CE        (%dx%d)       %7.2f ms %8.2f ms" % (seq, V, t_ce_f, t_ce_fb))
    print("  embedding gather         %7.2f ms %8.2f ms" % (t_emb_f, t_emb_fb))
    print("\n  head(fwd+bwd) + CE(fwd+bwd) = %.2f ms"
          % (t_head_fb + t_ce_f + t_ce_fb))
    print("  对照：目标书的固定开销是 **193 ms**（4 层拟合出来的）")


if __name__ == "__main__":
    main()
