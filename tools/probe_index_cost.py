# -*- coding: utf-8 -*-
"""查 `W3d[e]` 是否每次拷贝，以及 transpose 视图是否拷贝（每次清池单测）。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    E, dff, d = 8, 2048, 512
    rng = np.random.default_rng(0)
    W = T.tensor(rng.normal(0, .02, (E, dff, d)).astype(np.float32), requires_grad=True)
    flat = T.tensor(rng.normal(0, .02, (dff, d)).astype(np.float32), requires_grad=True)

    def m(name, fn):
        rt.empty_cache()
        p0 = rt.pool_stats()["peak_live_bytes"]
        y = fn()
        p1 = rt.pool_stats()["peak_live_bytes"]
        print("%-28s peak_live=%.0f MiB" % (name, (p1 - p0) / 2**20))
        del y
        rt.empty_cache()

    m("W[0] 单次索引", lambda: W[0])
    m("W[0] 连取 8 次", lambda: T.cat([W[e] for e in range(E)], dim=0))
    m("flat.transpose(0,1)", lambda: flat.transpose(0, 1))
    m("matmul x@flat.T", lambda: T.tensor(np.zeros((2048, d), np.float32)) @ flat.transpose(0, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
