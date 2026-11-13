# -*- coding: utf-8 -*-
"""逐算子记录**调用后仍留存**的 live（保留引用，模拟训练步里的图）。

目的：找出 V3 一层里 392 MiB 的留存落在哪些算子。
N=2048, d=512, d_ff=2048, E=8。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    N, d, dff, E = 2048, 512, 2048, 8
    rng = np.random.default_rng(0)
    W = lambda *s: T.tensor(rng.normal(0, .02, s).astype(np.float32), requires_grad=True)
    ln1, ln2 = W(d), W(d)
    wq, wk, wv = W(d, d), W(2 * 64, d), W(2 * 64, d)   # n_q=8, n_kv=2, d_k=64
    wo = W(d, d)
    router = W(E, d)
    W1 = [W(dff, d) for _ in range(E)]; W2 = [W(dff, d) for _ in range(E)]; W3 = [W(d, dff) for _ in range(E)]
    cos, sin = V.rope_cos_sin(N, 64, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    x0 = T.tensor(rng.normal(0, .02, (1, N, d)).astype(np.float32), requires_grad=True)

    rt.empty_cache()
    prev = rt.pool_stats()["live_bytes"]

    def step(name, fn):
        nonlocal prev
        y = fn()
        cur = rt.pool_stats()["live_bytes"]
        print("%-26s 留存 %+6.0f MiB（累计 %.0f）"
              % (name, (cur - prev) / 2**20, cur / 2**20))
        prev = cur
        return y

    a = step("rmsnorm(ln1)", lambda: V.rmsnorm_orv(x0, ln1))
    b = step("attention_orv(flash)", lambda: V.attention_orv(a, wq, wk, wv, wo, cos, sin, 64, 8, 2, flash=True))
    h = step("残差 x+attn", lambda: x0 + b)
    hn3 = step("rmsnorm(ln2)", lambda: V.rmsnorm_orv(h, ln2))
    hn = step("reshape", lambda: hn3.reshape(N, d))
    eids_g = step("route_top1", lambda: MOE.route_top1(hn, router))
    mo = step("moe_dispatch_orv", lambda: MOE.moe_dispatch_orv(hn, eids_g[0], eids_g[1], W1, W2, W3))
    y = step("残差 h+moe", lambda: h + mo.reshape(1, N, d))
    print("总留存 %.0f MiB" % (rt.pool_stats()["live_bytes"] / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
