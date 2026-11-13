# -*- coding: utf-8 -*-
"""在 V100 上验 `scatter_add` 的前向/反向（对着 numpy 判定基准）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import v3_scatter as S  # noqa: E402


def np_(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    n_table, n_vals, d = 5, 7, 3
    ids = rng.integers(0, n_table, n_vals).astype(np.int64)
    vals = rng.normal(size=(n_vals, d)).astype(np.float32)

    # ---- 前向 ----
    v = T.tensor(vals, requires_grad=True)
    out = S.scatter_add(v, ids, n_table)
    got = np_(out)
    want = V.scatter_add_numpy(ids, vals, n_table)
    print("前向 shape", got.shape, "max_err=%.3e" % float(np.abs(got - want).max()))

    # ---- 反向（sum 作 loss：values.grad 应全 1）----
    out.sum().backward()
    g = np_(v.grad)
    print("反向 max_err=%.3e（期望全 1）" % float(np.abs(g - 1.0).max()))

    # ---- 加权反向：与 numpy 的 gather 公式对拍 ----
    up = rng.normal(size=(n_table, d)).astype(np.float32)
    v2 = T.tensor(vals, requires_grad=True)
    out2 = S.scatter_add(v2, ids, n_table)
    (out2 * T.tensor(up)).sum().backward()
    g2 = np_(v2.grad)
    want2 = V.scatter_add_backward_numpy(ids, up)
    print("加权反向 max_err=%.3e" % float(np.abs(g2 - want2).max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
