# -*- coding: utf-8 -*-
"""打印 MoE 分派里各量形状，定位广播错。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from tgai_go.train import v3_ops as V  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    N, d, d_ff, E = 12, 8, 16, 4
    x = rng.normal(size=(N, d)).astype(np.float32)
    W1 = rng.normal(0, .2, (E, d_ff, d)).astype(np.float32)
    W2 = rng.normal(0, .2, (E, d_ff, d)).astype(np.float32)
    W3 = rng.normal(0, .2, (d, d_ff)).astype(np.float32)
    eids = rng.integers(0, E, N).astype(np.int64)
    gate = rng.random(N).astype(np.float32)

    print("x", x.shape, "W1", W1.shape, "W3", W3.shape, "eids", eids.shape, "gate", gate.shape)
    g1 = np.asarray(gate, np.float32).reshape(-1)
    print("g1", g1.shape)
    for e in range(E):
        pos = np.flatnonzero(eids == e)
        if pos.size == 0:
            continue
        y = V.situ_glu_numpy(x[pos], W1[e], W2[e], W3[e])
        print("e=%d pos=%s y=%s gate[pos]=%s" % (e, pos.shape, y.shape, g1[pos].shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
