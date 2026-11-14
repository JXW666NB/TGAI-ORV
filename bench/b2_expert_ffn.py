# -*- coding: utf-8 -*-
"""b2：专家 FFN（SiTU-GLU）单独基准：三种规模 × 前向/前+反向，并给等效算力。

专家 FFN 三个 GEMM：W1/W2 是 (d_ff, d)，W3 是 (d, d_ff)。
FLOPs（前向）≈ 2 * M * d * d_ff * 3；反向约 2 倍。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")
sys.path.insert(0, "D:/orv_pip_0962/bench")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402
from btimer import Timer, rnd  # noqa: E402


def main():
    T.set_default_device("cuda")
    tm = Timer()
    d, d_ff = 512, 2048
    W1 = T.tensor(rnd(d_ff, d, seed=1, scale=0.05))
    W2 = T.tensor(rnd(d_ff, d, seed=2, scale=0.05))
    W3 = T.tensor(rnd(d, d_ff, seed=3, scale=0.05))

    print("=== situ_glu_orv 前向（单专家）===")
    for M in (128, 256, 512, 1024, 2048):
        x = T.tensor(rnd(M, d, seed=M))
        flops = 2 * M * d * d_ff * 3
        tm.report("M=%-5d 前向" % M, lambda: V.situ_glu_orv(x, W1, W2, W3),
                  flops=flops)

    print("=== situ_glu_orv 前+反向（单专家）===")
    for M in (128, 512, 2048):
        x = T.tensor(rnd(M, d, seed=M), requires_grad=True)

        def fwbwd():
            y = V.situ_glu_orv(x, W1, W2, W3)
            (y * y).sum().backward()
        flops = 2 * M * d * d_ff * 3 * 3      # 前向 1 + 反向 2
        tm.report("M=%-5d 前+反向" % M, fwbwd, flops=flops, reps=10)

    print("=== 同规模多次调用（看主机成本是否被摊薄）===")
    x = T.tensor(rnd(256, d, seed=7))

    def eight():
        for _ in range(8):
            V.situ_glu_orv(x, W1, W2, W3)
    tm.report("8 次 M=256 前向", eight, flops=2 * 256 * d * d_ff * 3 * 8, reps=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
