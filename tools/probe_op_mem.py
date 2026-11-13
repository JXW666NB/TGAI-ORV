# -*- coding: utf-8 -*-
"""逐算子、每次清池，测各步分配的峰值（N=2048, d=512, d_ff=2048）。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    N, d, dff = 2048, 512, 2048
    rng = np.random.default_rng(0)
    x = T.tensor(rng.normal(0, .02, (N, d)).astype(np.float32), requires_grad=True)
    w1 = T.tensor(rng.normal(0, .02, (dff, d)).astype(np.float32))
    big = T.tensor(rng.normal(0, .02, (N, dff)).astype(np.float32), requires_grad=True)

    def m(name, fn):
        rt.empty_cache()
        pk0 = rt.pool_stats()["peak_live_bytes"]
        y = fn()
        pk = rt.pool_stats()["peak_live_bytes"]
        print("%-24s peak_live=%.0f MiB" % (name, (pk - pk0) / 2**20))
        del y
        rt.empty_cache()

    m("空操作（基线）", lambda: x * 1.0)
    m("x @ w1.T  (N,512)x(512,dff)", lambda: x @ w1.transpose(0, 1))
    m("big * 0.25  (N,dff)", lambda: big * 0.25)
    m("big.tanh()", lambda: big.tanh())
    m("big * big", lambda: big * big)
    m("big @ w3.T 反方向", lambda: T.tensor(np.eye(dff, dtype=np.float32)) @ big.transpose(0, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
