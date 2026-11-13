# -*- coding: utf-8 -*-
"""按 seq 扫，逐算子记录**留存**，找出哪一项随 seq 超线性增长。

复现用户观察：ORV 显存随 seq 近似指数，PyT 每档只涨 ~100 MB。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402


def run_seq(N, d, dff, E):
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    W = lambda *s: T.tensor(rng.normal(0, .02, s).astype(np.float32), requires_grad=True)
    ln1, ln2 = W(d), W(d)
    wq, wk, wv, wo = W(d, d), W(2 * 64, d), W(2 * 64, d), W(d, d)
    router = W(E, d)
    W1 = [W(dff, d) for _ in range(E)]
    W2 = [W(dff, d) for _ in range(E)]
    W3 = [W(d, dff) for _ in range(E)]
    cos, sin = V.rope_cos_sin(N, 64, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    x0 = T.tensor(rng.normal(0, .02, (1, N, d)).astype(np.float32), requires_grad=True)

    rt.empty_cache()
    prev = rt.pool_stats()["live_bytes"]
    acc = {}

    def step(name, fn):
        nonlocal prev
        y = fn()
        cur = rt.pool_stats()["live_bytes"]
        acc[name] = (cur - prev) / 2**20
        prev = cur
        return y

    a = step("rmsnorm", lambda: V.rmsnorm_orv(x0, ln1))
    b = step("attn", lambda: V.attention_orv(a, wq, wk, wv, wo, cos, sin, 64, 8, 2, flash=True))
    h = step("res1", lambda: x0 + b)
    hn3 = step("rmsnorm2", lambda: V.rmsnorm_orv(h, ln2))
    hn = step("reshape", lambda: hn3.reshape(N, d))
    eg = step("route", lambda: MOE.route_top1(hn, router))
    mo = step("moe", lambda: MOE.moe_dispatch_orv(hn, eg[0], eg[1], W1, W2, W3))
    step("res2", lambda: h + mo.reshape(1, N, d))
    return acc


def main():
    print("%-10s %9s %9s %9s %9s %9s %9s" % ("seq", "rmsnorm", "attn", "rms2", "route", "moe", "合计"))
    for N in (512, 1024, 2048):
        a = run_seq(N, 512, 2048, 8)
        tot = sum(a.values())
        print("%-10d %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f"
              % (N, a["rmsnorm"], a["attn"], a["rmsnorm2"], a["route"], a["moe"], tot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
