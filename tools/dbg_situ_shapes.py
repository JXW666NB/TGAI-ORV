# -*- coding: utf-8 -*-
"""逐量打印 situ_glu_numpy 的中间形状，定位 (2,) 的来源。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tgai_go.train import v3_ops as V  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    N, d, d_ff = 12, 8, 16
    x = rng.normal(size=(N, d)).astype(np.float32)
    W1 = rng.normal(0, .2, (4, d_ff, d)).astype(np.float32)
    W2 = rng.normal(0, .2, (4, d_ff, d)).astype(np.float32)
    W3 = rng.normal(0, .2, (d, d_ff)).astype(np.float32)
    pos = np.array([3, 7])

    xs = x[pos]
    print("xs", xs.shape, "W1[0]", W1[0].shape, "W3", W3.shape, "W3.T", W3.T.shape)
    g = xs @ W1[0].T
    u = xs @ W2[0].T
    print("g", g.shape, "u", u.shape)
    gate = (4.0 * np.tanh(g / 4.0)) * (0.5 * (1.0 + np.tanh(0.5 * g)))
    up = 25.0 * np.tanh(u / 25.0)
    print("gate", gate.shape, "up", up.shape, "prod", (gate * up).shape)
    y1 = (gate * up) @ W3.T
    print("手工 y", y1.shape)
    y2 = V.situ_glu_numpy(xs, W1[0], W2[0], W3)
    print("函数 y", np.asarray(y2).shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
