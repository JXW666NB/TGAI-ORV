# -*- coding: utf-8 -*-
"""把一次 MoE 分派拆开计时：gather / 每专家 GEMM+SiTU / scatter / 反向。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_moe as MOE, v3_ops as V  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    N, D, E, DFF, K = 2048, 512, 8, 2048, 2
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, .3, (N, D)).astype(np.float32)
    router = T.tensor(rng.normal(0, .3, (E, D)).astype(np.float32))
    W1 = [T.tensor(rng.normal(0, .05, (DFF, D)).astype(np.float32)) for _ in range(E)]
    W2 = [T.tensor(rng.normal(0, .05, (DFF, D)).astype(np.float32)) for _ in range(E)]
    W3 = [T.tensor(rng.normal(0, .05, (D, DFF)).astype(np.float32)) for _ in range(E)]

    def timeit(label, fn, reps=5):
        fn(); rt.sync()
        t0 = time.time()
        for _ in range(reps):
            r = fn()
        rt.sync()
        dt = (time.time() - t0) / reps
        print("%-34s %8.2f ms" % (label, dt * 1e3))
        return r

    tok = np.repeat(np.arange(N), K)
    eids = rng.integers(0, E, N * K).astype(np.int64)
    order = np.argsort(eids, kind="stable")

    x = T.tensor(x0)
    timeit("x[tok]  gather 4096 行", lambda: x[tok])
    timeit("x[order] gather 4096 行", lambda: x[order])

    # 单专家：M=512 行
    xs = T.tensor(rng.normal(0, .3, (512, D)).astype(np.float32), requires_grad=True)
    y = timeit("单专家 SiTU-GLU 前向(M=512)",
               lambda: V.situ_glu_orv(xs, W1[0], W2[0], W3[0]))
    y2 = V.situ_glu_orv(xs, W1[0], W2[0], W3[0])
    def bwd():
        l = (y2 * y2).sum()
        l.backward()
    timeit("单专家 SiTU-GLU 前向+反向", bwd)

    # 整次分派
    xe = T.tensor(x0, requires_grad=True)
    ee, gg, _ = MOE.route_topk(xe, router, K)
    def full():
        m = MOE.moe_topk_orv(xe, ee, gg, W1, W2, W3)
        (m * m).sum().backward()
    timeit("整次分派 前向+反向(合并)", full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
