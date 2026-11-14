# -*- coding: utf-8 -*-
"""把一步训练按部件计时：8 层、seq2048、8 头、top_k=2、共享专家。

目的是给"0.50 s/步"一个可信的分解，而不是从单层外推。
每步各部件累计（8 层求和），再除以步数。
"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402
from v3_orv_train import build, params_of  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    d, S, E, L, n_q, n_kv = 512, 2048, 8, 8, 8, 2
    d_k = d // n_q
    cfg = dict(d=d, d_ff=4 * d, E=E, layers=L, n_q=n_q, n_kv=n_kv, vocab=4096,
               seq=S, steps=1)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    cos, sin = V.rope_cos_sin(S, d_k, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    opt = T.optim.AdamW(params_of(P), lr=3e-4)
    ids = rng.integers(0, 4096, S).astype(np.int64)
    tgt = np.roll(ids, -1)

    acc = {}

    def tick(name, fn):
        # GPU 是异步的：不同步就只会量到"入队时间"。这里显式同步，接受额外开销。
        rt.sync()
        t0 = time.time()
        r = fn()
        rt.sync()
        acc[name] = acc.get(name, 0.0) + (time.time() - t0)
        return r

    steps = 3
    for step in range(steps):
        for k in list(acc):
            acc[k] = 0.0
        opt.zero_grad()
        x = tick("embed", lambda: P["emb"][ids].reshape(1, S, d))
        for Li, Lw in enumerate(P["layers"]):
            a = tick("rmsnorm", lambda Lw=Lw: V.rmsnorm_orv(x, Lw["ln1"]))
            at = tick("attention", lambda a=a, Lw=Lw: V.attention_orv(
                a, Lw["wq"], Lw["wk"], Lw["wv"], Lw["wo"], cos, sin, d_k, n_q, n_kv))
            h = tick("residual", lambda at=at: x + at)
            n2 = tick("rmsnorm", lambda h=h, Lw=Lw: V.rmsnorm_orv(h, Lw["ln2"]))
            hn = n2.reshape(S, d)
            r = tick("route", lambda hn=hn, Lw=Lw: MOE.route_topk(hn, Lw["router"], 2))
            mo = tick("moe", lambda r=r, hn=hn, Lw=Lw: MOE.moe_topk_orv(
                hn, r[0], r[1], Lw["W1"], Lw["W2"], Lw["W3"]))
            mo = tick("shared", lambda mo=mo, hn=hn, Lw=Lw: mo + V.situ_glu_orv(
                hn, Lw["S1"], Lw["S2"], Lw["S3"]))
            x = tick("residual", lambda h=h, mo=mo: h + mo.reshape(1, S, d))
        lg = tick("head", lambda: V.rmsnorm_orv(x, P["ln_f"]).reshape(S, d) @ P["head"].transpose(0, 1))
        loss = tick("ce", lambda: T.functional.cross_entropy(lg, tgt))
        tick("backward", lambda: loss.backward())
        tick("adam", lambda: opt.step())
        del x, lg, loss
        rt.empty_cache()
        if step == steps - 1:
            total = sum(acc.values())
            print("一步总计 %.3f s（各部件 8 层累计）" % total)
            for k in sorted(acc, key=lambda z: -acc[z]):
                print("  %-10s %.3f s  %4.1f%%" % (k, acc[k], 100 * acc[k] / total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
