# -*- coding: utf-8 -*-
"""拆 moe_dispatch_orv 内部：xs / 逐专家 / cat / 门控 / scatter 各留多少。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE, v3_scatter as S  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    N, d, dff, E = 2048, 512, 2048, 8
    rng = np.random.default_rng(0)
    W = lambda *s: T.tensor(rng.normal(0, .02, s).astype(np.float32), requires_grad=True)
    W1 = [W(dff, d) for _ in range(E)]
    W2 = [W(dff, d) for _ in range(E)]
    W3 = [W(d, dff) for _ in range(E)]
    router = W(E, d)
    hn = T.tensor(rng.normal(0, .02, (N, d)).astype(np.float32), requires_grad=True)

    rt.empty_cache()
    prev = rt.pool_stats()["live_bytes"]

    def step(name, fn):
        nonlocal prev
        y = fn()
        cur = rt.pool_stats()["live_bytes"]
        print("%-22s 留存 %+6.0f MiB" % (name, (cur - prev) / 2**20))
        prev = cur
        return y

    eids, g, _s = step("route_top1", lambda: MOE.route_top1(hn, router))
    order = np.argsort(eids, kind="stable")
    xs = step("xs = x[order]", lambda: hn[order])
    gs = step("gate[order]", lambda: g[order])
    parts = []
    for e in range(E):
        pos = (eids == e)
        cnt = int(pos.sum())
        if cnt == 0:
            continue
        seg = xs[int(np.flatnonzero(np.cumsum(np.bincount(eids, minlength=E))[:e].sum() if e else 0)):] if False else None
        parts.append(V.situ_glu_orv(xs[len(parts) * (N // E):(len(parts) + 1) * (N // E)], W1[e], W2[e], W3[e]))
    step("8 个专家 situ_glu", lambda: parts)
    y_all = step("cat(parts)", lambda: T.cat(parts, dim=0))
    scaled = step("y * gate", lambda: y_all * gs.reshape(-1, 1))
    out = step("scatter_add", lambda: S.scatter_add(scaled, order, N))
    print("总留存 %.0f MiB" % (rt.pool_stats()["live_bytes"] / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
