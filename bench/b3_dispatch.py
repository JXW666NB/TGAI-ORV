# -*- coding: utf-8 -*-
"""b3：MoE 分派逐相位基准（真实 GPU 流时间）。

每个相位先打印名字再执行，且单独捕获异常 —— 一次运行就能看清在哪一段炸/慢。
"""
import sys
import traceback

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
    print("counts =", counts.tolist(), flush=True)

    x = T.tensor(rnd(N, d, seed=1))
    gate = T.tensor(rng.uniform(0.1, 1.0, (N * K,)).astype(np.float32))
    rows_bytes = N * K * d * 4

    def safe(label, fn, **kw):
        print("→ " + label, flush=True)
        try:
            return tm.report(label, fn, **kw)
        except Exception:
            traceback.print_exc()
            return None

    safe("① x[tok] 展开", lambda: x[tok], bytes_moved=rows_bytes)

    x_rows = T.tensor(rnd(N * K, d, seed=2))
    safe("② index_select 按专家", lambda: IDX.index_select(x_rows, 0, order),
         bytes_moved=rows_bytes)
    xs = IDX.index_select(x_rows, 0, order)
    g_sorted = gate[order]

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
    safe("③ 8 专家 FFN", experts, flops=flops_experts)

    parts = experts()
    safe("④ cat 8 份", lambda: T.cat(parts, dim=0), bytes_moved=rows_bytes)
    ys = T.cat(parts, dim=0)
    safe("⑤ scatter_add 散回", lambda: S.scatter_add(
        ys * g_sorted.reshape(-1, 1), order, N * K), bytes_moved=rows_bytes)

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
    safe("⑥ 整条分派 前向", full_fwd, flops=flops_experts, reps=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
