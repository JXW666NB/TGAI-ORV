# -*- coding: utf-8 -*-
"""单独计量 MoE 各步的留存：situ_glu / gather / cat / scatter_add（N=2048, d=512, d_ff=2048）。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import v3_scatter as S  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    N, d, d_ff = 2048, 512, 2048
    rng = np.random.default_rng(0)
    x = T.tensor(rng.normal(0, .02, (N, d)).astype(np.float32), requires_grad=True)
    w1 = T.tensor(rng.normal(0, .02, (d_ff, d)).astype(np.float32))
    w2 = T.tensor(rng.normal(0, .02, (d_ff, d)).astype(np.float32))
    w3 = T.tensor(rng.normal(0, .02, (d, d_ff)).astype(np.float32))
    ids = rng.permutation(N).astype(np.int64)

    def m(name, fn):
        rt.empty_cache()
        pk0 = rt.pool_stats()["peak_live_bytes"]
        y = fn()
        pk = rt.pool_stats()["peak_live_bytes"]
        print("%-22s peak_live=%.0f MiB（增量 %.0f）"
              % (name, pk / 2**20, (pk - pk0) / 2**20))
        del y
        rt.empty_cache()

    m("situ_glu_orv 全量", lambda: V.situ_glu_orv(x, w1, w2, w3))
    xs = x[ids]
    m("gather x[ids]", lambda: x[ids])
    m("cat 8 段", lambda: T.cat([xs[i * 256:(i + 1) * 256] for i in range(8)], dim=0))
    m("scatter_add", lambda: S.scatter_add(xs, ids, N))
    m("relu/exp 单项", lambda: xs.exp())

    st = rt.pool_stats()
    print("结束时 live=%.0f MiB" % (st["live_bytes"] / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
