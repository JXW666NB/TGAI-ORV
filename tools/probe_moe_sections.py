# -*- coding: utf-8 -*-
"""把 MoE 分派的 6 段分别测「前向+反向」，找出 62 ms 在哪一段。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE, v3_scatter as S  # noqa: E402


def bench(label, build, reps=10):
    """build() 返回 (loss_fn) —— 每次都新建图，避免复用已释放的中间量。"""
    f = build()
    f(); T._cuda().sync()                       # 预热
    T._cuda().empty_cache()
    t0 = time.time()
    for _ in range(reps):
        f = build()
        f()
    T._cuda().sync()
    dt = (time.time() - t0) / reps
    print("%-38s %8.2f ms" % (label, dt * 1e3))
    return dt


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    N, D, E, DFF, K = 2048, 512, 8, 2048, 2
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, .3, (N, D)).astype(np.float32)
    W1 = [T.tensor(rng.normal(0, .05, (DFF, D)).astype(np.float32)) for _ in range(E)]
    W2 = [T.tensor(rng.normal(0, .05, (DFF, D)).astype(np.float32)) for _ in range(E)]
    W3 = [T.tensor(rng.normal(0, .05, (D, DFF)).astype(np.float32)) for _ in range(E)]
    eids = rng.integers(0, E, N * K).astype(np.int64)
    order = np.argsort(eids, kind="stable")
    gate0 = rng.uniform(0.1, 1.0, N * K).astype(np.float32)
    # 合并路径里 x 已经是 N*K 行（先按 token 展开），order 也在 [0, N*K)
    x4 = rng.normal(0, .3, (N * K, D)).astype(np.float32)

    def mk_gather():
        x = T.tensor(x4, requires_grad=True)
        def f():
            y = x[order]
            (y * y).sum().backward()
        return f

    def mk_gate():
        g = T.tensor(gate0, requires_grad=True)
        def f():
            y = g[order]
            (y * y).sum().backward()
        return f

    def mk_cat():
        parts = [T.tensor(x4[:512], requires_grad=True) for _ in range(E)]
        def f():
            y = T.cat(parts, dim=0)
            (y * y).sum().backward()
        return f

    def mk_situ():
        rows = T.tensor(x0[:256], requires_grad=True)
        def f():
            y = V.situ_glu_orv(rows, W1[0], W2[0], W3[0])
            (y * y).sum().backward()
        return f

    def mk_situ8():
        def f():
            acc = None
            for e in range(E):
                r = T.tensor(x0[:256], requires_grad=True)
                y = V.situ_glu_orv(r, W1[e], W2[e], W3[e])
                l = (y * y).sum()
                l.backward()
                acc = l
        return f

    def mk_scatter():
        ys = T.tensor(x4, requires_grad=True)
        def f():
            y = S.scatter_add(ys, order, N * K)
            (y * y).sum().backward()
        return f

    bench("x[order]  gather fwd+bwd", mk_gather)
    bench("gate[order] fwd+bwd", mk_gate)
    bench("T.cat(8 份) fwd+bwd", mk_cat)
    bench("situ_glu 单专家(M=256) fwd+bwd", mk_situ)
    bench("situ_glu 8 专家(M=256×8) fwd+bwd", mk_situ8)
    bench("scatter_add(4096) fwd+bwd", mk_scatter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
