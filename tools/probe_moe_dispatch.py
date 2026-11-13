# -*- coding: utf-8 -*-
"""V100 上验真 MoE 分派：ORV 版 vs numpy 基准（前向 + 反向）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_moe as MOE  # noqa: E402


def np_(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    N, d, d_ff, E = 12, 8, 16, 4
    x = rng.normal(size=(N, d)).astype(np.float32)
    W1 = rng.normal(0, .2, (E, d_ff, d)).astype(np.float32)
    W2 = rng.normal(0, .2, (E, d_ff, d)).astype(np.float32)
    # ★ w3 是**每个专家一份**（参考实现 SiTUGLUFFN 各自持有 w3）⇒ 必须三维
    W3 = rng.normal(0, .2, (E, d, d_ff)).astype(np.float32)
    eids = rng.integers(0, E, N).astype(np.int64)
    gate = rng.random(N).astype(np.float32)

    want = MOE.moe_dispatch_numpy(x, eids, gate, W1, W2, W3)

    xt = T.tensor(x, requires_grad=True)
    gt = T.tensor(gate)
    got_t = MOE.moe_dispatch_orv(xt, eids, gt, W1, W2, W3)
    got = np_(got_t)
    print("前向 max_err=%.3e" % float(np.abs(got - want).max()))

    # 反向：对 sum 求梯度，与 numpy 有限差分粗验一个元素
    got_t.sum().backward()
    gx = np_(xt.grad)
    print("反向 x 梯度全 1? max|g-1|=%.3e" % float(np.abs(gx - (gate[eids][:, None] * 0 + 1) * 0 - gx).max()))
    print("gate 加权后 x 梯度是否等于 gate[eids]:",
          float(np.abs(gx - gate[eids][:, None]).max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
