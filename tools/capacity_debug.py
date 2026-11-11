# -*- coding: utf-8 -*-
"""插桩定位：CUDA 路径的容量截断为何没生效。

打印 `_capacity` 返回的 cap、截断前后的 off、以及每个专家的 token 数。

运行（需独占 GPU）：python capacity_debug.py
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TGAI-ORV-main", "python"))

D, F, E, K, N = 64, 128, 8, 2, 96
OUT = os.path.join(HERE, "cdbg.txt")


def main():
    import tgai_go.train as T
    rep = []
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (N, D)).astype(np.float32)

    T.set_default_device("cuda")
    rt = T._cuda()
    m = T.nn.MoE(D, F, E, top_k=K, gate="softmax_topk", capacity_factor=1.0)
    seed = np.random.default_rng(1)
    for p in m.parameters():
        p._gpu = rt.h2d(seed.normal(0, 0.05, tuple(int(s) for s in p.shape)).astype(np.float32))

    rep.append("capacity_factor=%r, cap(_capacity(%d))=%r"
               % (m.capacity_factor, N, m._capacity(N)))

    orig = m._clamp_offsets

    def traced(off, cap):
        out = orig(off, cap)
        rep.append("  off  before=%s" % np.asarray(off).tolist())
        rep.append("  off  after =%s" % np.asarray(out).tolist())
        rep.append("  counts before=%s"
                   % [int(off[i + 1]) - int(off[i]) for i in range(len(off) - 1)])
        rep.append("  counts after =%s"
                   % [int(out[i + 1]) - int(out[i]) for i in range(len(out) - 1)])
        return out

    m._clamp_offsets = traced
    y = np.asarray(m(T.tensor(X)).to_numpy(), np.float32)
    rep.append("GPU 输出整行为 0 的 token 数 = %d" % int((np.abs(y).max(axis=1) == 0).sum()))

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
