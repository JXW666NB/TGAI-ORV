# -*- coding: utf-8 -*-
"""b3：MoE 分派逐相位基准（真实 GPU 时间），前向与反向分别量。

相位：gather → 按专家 gather → 8 专家 FFN → cat → scatter → （反向）。
判据：把相位时间与理论最小值（FLOPs/带宽）比；差距以数量级计的相位就是病灶。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")
sys.path.insert(0, "D:/orv_pip_0962/bench")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import indexing as IDX  # noqa: E402
from tgai_go.train import v3_scatter as S  # noqa: E402
from btimer import Timer, rnd  # noqa: E402


def main():
    T.set_default_device("cuda")
    tm = Timer()
    N, d, d_ff, E, K = 2048, 512, 2048, 8, 2
    rng = np.random.default_rng(0)
    W1 = [T.tensor(rnd(d_ff, d, seed=10 + e, scale=0.05)) for e in range(E)]
    W2 = [T.tensor(rnd(d_ff, d, seed=20 + e, scale=0.05)) for e in range(E)]
    W3 = [T.tensor(rnd(d, d_ff, seed=30 + e, scale=0.05)) for e in range(E)]

    eids = rng.integers(0, E, (N, K)).astype(np.int64)
    e_flat = eids.reshape(-1)
    order = np.argsort(e_flat, kind="stable")
    counts = np.bincount(e_flat, minlength=E)
    bounds = np.cumsum(counts)
    tok = np.repeat(np.arange(N), K)

    x = T.tensor(rnd(N, d, seed=1))
    gate = T.tensor(rng.uniform(0.1, 1.0, (N * K,)).astype(np.float32))
    x_rows = IDX.index_select(x, 0, tok) if hasattr(IDX, "index_select") else x[tok]
    xs = IDX.index_select(x_rows, 0, order)
    g_sorted = gate[order]

    rows_bytes = N * K * d * 4

    print("=== 前向相位（真实 GPU 时间）===")
    tm.report("① x[tok] 展开", lambda: x[tok], bytes_moved=rows_bytes)
    tm.report("② index_select 按专家", lambda: IDX.index_select(x_rows, 0, order),
              bytes_moved=rows_bytes)

    parts = []
    start = 0
    for e in range(E):
        end = int(bounds[e])
        if end > start:
            parts.append(V.situ_glu_orv(xs[start:end], W1[e], W2[e], W3[e]))
        start = end

    def experts():
        out = []
        st = 0
        for e in range(E):
            en = int(bounds[e])
            if en > st:
                out.append(V.situ_glu_orv(xs[st:en], W1[e], W2[e], W3[e]))
            st = en
        return out
    flops_experts = 2 * N * K * d * d_ff * 3
    tm.report("③ 8 专家 FFN", experts, flops=flops_experts)

    tm.report("④ cat 8 份", lambda: T.cat(parts, dim=0), bytes_moved=rows_bytes)
    ys = T.cat(parts, dim=0)
    tm.report("⑤ scatter_add 散回", lambda: S.scatter_add(
        ys * g_sorted.reshape(-1, 1), order, N * K), bytes_moved=rows_bytes)

    print("=== 整条前向（串起来）===")

    def full_fwd():
        xr = x[tok]
        xs2 = IDX.index_select(xr, 0, order)
        gs2 = gate[order]
        ps = []
        st = 0
        for e in range(E):
            en = int(bounds[e])
            if en > st:
                ps.append(V.situ_glu_orv(xs2[st:en], W1[e], W2[e], W3[e]))
            st = en
        y2 = T.cat(ps, dim=0)
        return S.scatter_add(y2 * gs2.reshape(-1, 1), order, N * K)
    tm.report("整条分派 前向", full_fwd, flops=flops_experts, reps=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
