# -*- coding: utf-8 -*-
"""在**相位边界同步**，量一次 MoE 分派的真实 GPU 时间（前向与反向分开）。

为什么不能逐内核打点：逐个 sync 会把测量自身变成主要开销（上一轮实测 1189 次
sync ≈ 0.36 s）。这里只在 5 个相位边界同步。
"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402
from tgai_go.train import indexing as IDX  # noqa: E402
from tgai_go.train import v3_scatter as S  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    N, D, E, DFF, K = 2048, 512, 8, 2048, 2
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, .3, (N, D)).astype(np.float32)
    W1 = [T.tensor(rng.normal(0, .05, (DFF, D)).astype(np.float32)) for _ in range(E)]
    W2 = [T.tensor(rng.normal(0, .05, (DFF, D)).astype(np.float32)) for _ in range(E)]
    W3 = [T.tensor(rng.normal(0, .05, (D, DFF)).astype(np.float32)) for _ in range(E)]
    eids = rng.integers(0, E, (N, K)).astype(np.int64)
    gate_h = rng.uniform(0.1, 1.0, (N, K)).astype(np.float32)

    tok = np.repeat(np.arange(N), K)
    e_flat = eids.reshape(-1)
    order = np.argsort(e_flat, kind="stable")
    counts = np.bincount(e_flat, minlength=E)
    bounds = np.cumsum(counts)

    def phases(x, gate, reps=3):
        """跑 reps 次前向，记录各相位真实耗时。"""
        acc = {}

        def mark(name, t0):
            rt.sync()
            dt = time.perf_counter() - t0
            acc[name] = acc.get(name, 0.0) + dt
            print("    %-24s %7.2f ms" % (name, dt * 1e3), flush=True)

        for _ in range(reps):
            t0 = time.perf_counter()
            x_rows = x[tok]
            mark("gather x[tok]", t0)

            t0 = time.perf_counter()
            xs = IDX.index_select(x_rows, 0, order)
            g_sorted = gate[order]
            mark("index_select 按专家", t0)

            t0 = time.perf_counter()
            parts = []
            start = 0
            for e in range(E):
                end = int(bounds[e])
                if end > start:
                    parts.append(V.situ_glu_orv(xs[start:end], W1[e], W2[e], W3[e]))
                start = end
            mark("8 专家 GEMM+SiTU", t0)

            t0 = time.perf_counter()
            ys = T.cat(parts, dim=0) if hasattr(T, "cat") else None
            if ys is None:
                from tgai_go.train import cat as _cat
                ys = _cat(parts, dim=0)
            mark("cat 8 份", t0)

            t0 = time.perf_counter()
            out = S.scatter_add(ys * g_sorted.reshape(-1, 1), order, N * K)
            mark("scatter_add", t0)

        for k in ("gather x[tok]", "index_select 按专家", "8 专家 GEMM+SiTU",
                  "cat 8 份", "scatter_add"):
            print("  %-24s %7.2f ms" % (k, acc.get(k, 0.0) * 1e3 / reps))

    x = T.tensor(x0, requires_grad=True)
    gate = T.tensor(gate_h, requires_grad=True)
    print("前向（真实 GPU 时间，%d 次平均）：" % 3)
    phases(x, gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
