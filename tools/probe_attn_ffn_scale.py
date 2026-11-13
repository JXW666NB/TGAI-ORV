# -*- coding: utf-8 -*-
"""分离注意力与 FFN 的反向开销随 seq 的走势（1 层，E=1）。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402


def one(N, kind):
    rt = T._cuda()
    d, dff, E = 512, 2048, 1
    rng = np.random.default_rng(0)
    W = lambda *s: T.tensor(rng.normal(0, .02, s).astype(np.float32), requires_grad=True)
    ln = W(d)
    wq, wk, wv, wo = W(d, d), W(128, d), W(128, d), W(d, d)
    router = W(E, d)
    W1, W2, W3 = [W(dff, d)], [W(dff, d)], [W(d, dff)]
    cos, sin = V.rope_cos_sin(N, 64, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    x = T.tensor(rng.normal(0, .02, (1, N, d)).astype(np.float32), requires_grad=True)

    rt.empty_cache()
    p0 = rt.pool_stats()["peak_live_bytes"]
    if kind == "attn":
        y = x + V.attention_orv(V.rmsnorm_orv(x, ln), wq, wk, wv, wo, cos, sin, 64, 8, 2, flash=True)
    else:
        hn = V.rmsnorm_orv(x, ln).reshape(N, d)
        eg = MOE.route_top1(hn, router)
        mo = MOE.moe_dispatch_orv(hn, eg[0], eg[1], W1, W2, W3)
        y = x + mo.reshape(1, N, d)
    loss = y.sum()
    loss.backward()
    p1 = rt.pool_stats()["peak_live_bytes"]
    print("N=%-5d %-5s peak_live=%6.0f MiB" % (N, kind, (p1 - p0) / 2**20))
    del x, y, loss
    rt.empty_cache()


def main():
    T.set_default_device("cuda")
    for N in (512, 1024, 2048):
        for kind in ("attn", "ffn"):
            one(N, kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
