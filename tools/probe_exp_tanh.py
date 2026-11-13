# -*- coding: utf-8 -*-
"""比较 exp / tanh / log 的单位开销（同形状 16 MB 张量，每次清池）。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    a = np.random.rand(2048, 2048).astype(np.float32) * 0.5 + 0.1
    b = T.tensor(a, requires_grad=True)

    def m(name, fn):
        rt.empty_cache()
        p0 = rt.pool_stats()["peak_live_bytes"]
        y = fn()
        p1 = rt.pool_stats()["peak_live_bytes"]
        print("%-10s peak_live=%.0f MiB" % (name, (p1 - p0) / 2 ** 20))
        del y
        rt.empty_cache()

    m("exp", lambda: b.exp())
    m("tanh", lambda: b.tanh())
    m("log", lambda: b.log())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
