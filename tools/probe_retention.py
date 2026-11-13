# -*- coding: utf-8 -*-
"""归因：重复 单层 fwd/backward，看活跃显存是否随迭代增长。

假设：检查点版本里"每层重算的图没被释放"，导致 8 层累积。
若是，则 `live_bytes` 会随迭代线性上升；若平稳，则问题在别处。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402
from v3_orv_train import build  # noqa: E402
from v3_orv_train_ckpt import layer_fwd, leaf  # noqa: E402


def main():
    T.set_default_device("cuda")
    d, seq, n_q, n_kv, E = 512, 2048, 8, 2, 8
    cfg = dict(d=d, d_ff=4 * d, E=E, layers=1, n_q=n_q, n_kv=n_kv,
               vocab=4096, seq=seq, steps=1)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    L = P["layers"][0]
    d_k = d // n_q
    cos, sin = V.rope_cos_sin(seq, d_k, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    x0 = rng.normal(0, .02, (1, seq, d)).astype(np.float32)

    rt = T._cuda()
    print("iter  live_MiB  peak_live_MiB  occupied_MiB")
    for it in range(8):
        xi = leaf(x0)
        y = layer_fwd(xi, L, cos, sin, d_k, n_q, n_kv)
        y.backward(T.tensor(np.ones((1, seq, d), np.float32)))
        st = rt.pool_stats()
        print("%4d  %9.0f  %13.0f  %12.0f"
              % (it, st["live_bytes"] / 2**20, st["peak_live_bytes"] / 2**20,
                 st["occupied_bytes"] / 2**20))
        rt.empty_cache()
    st = rt.pool_stats()
    print("最终 peak_live_MiB = %.0f（若随迭代线性增长 ⇒ 图没释放）"
          % (st["peak_live_bytes"] / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
