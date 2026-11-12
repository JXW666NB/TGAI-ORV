# -*- coding: utf-8 -*-
"""单次测量：给定 (L, S, B, ckpt)，打印前向活张量 / 反向峰值 / 反向结束后。

**必须每个配置单独起进程**：`peak_live_bytes` 是进程级高水位、从不重置，
在同一进程里连跑多组会把前一组的峰值算进来（我为此误判过一次）。

由 `ckpt_scale.py` 以子进程方式调用。
"""
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_pkg_root(start):
    cur = start
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, "python", "tgai_go")):
            return os.path.join(cur, "python")
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    raise SystemExit("找不到 python/tgai_go")


sys.path.insert(0, _find_pkg_root(HERE))

D, NH, FF = 1280, 10, 5120
DH = D // NH


def main():
    L = int(sys.argv[1])
    S = int(sys.argv[2])
    B = int(sys.argv[3])
    ckpt = int(sys.argv[4]) == 1

    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)

    def dev(n, m):
        return T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.02, (n, m)).astype(np.float32)))

    qkv = [dev(3 * D, D) for _ in range(L)]
    o = [dev(D, D) for _ in range(L)]
    gate = [dev(FF, D) for _ in range(L)]
    up = [dev(FF, D) for _ in range(L)]
    down = [dev(D, FF) for _ in range(L)]
    allp = qkv + o + gate + up + down
    for t in allp:
        t.requires_grad = True

    x0 = T.Tensor._from_gpu(rt.h2d(
        rng.normal(0, 0.02, (B * S, D)).astype(np.float32))).reshape(B, S, D)
    x0.requires_grad = True

    def block(x, i):
        Bv, Sq, _ = x.shape
        z = x.reshape(Bv * Sq, D) @ qkv[i].T
        q, k, v = z.reshape(Bv, Sq, 3, NH, DH).permute(2, 0, 3, 1, 4)
        q = q.reshape(Bv * NH, Sq, DH)
        k = k.reshape(Bv * NH, Sq, DH)
        v = v.reshape(Bv * NH, Sq, DH)
        y = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)
        y = y.reshape(Bv, NH, Sq, DH).permute(0, 2, 1, 3).reshape(Bv * Sq, D)
        x = x + (y @ o[i].T).reshape(Bv, Sq, D)
        f = x.reshape(Bv * Sq, D)
        h = Fn.silu(f @ gate[i].T) * (f @ up[i].T)
        h = h @ down[i].T
        return x + h.reshape(Bv, Sq, D)

    st0 = rt.pool_stats()
    base = st0["live_bytes"]
    res = dict(L=L, S=S, B=B, ckpt=ckpt,
               weights_gib=sum(int(np.prod(tuple(t.shape))) for t in allp)
               * 4 / (1 << 30))

    x = x0
    for i in range(L):
        if ckpt:
            x = T.checkpoint(lambda h, _i=i: block(h, _i), x)
        else:
            x = block(x, i)
    res["fwd_live_gib"] = (rt.pool_stats()["live_bytes"] - base) / (1 << 30)

    try:
        loss = (x.reshape(-1, D) * x.reshape(-1, D)).sum()
        loss.backward()
        rt.sync()
        st = rt.pool_stats()
        res["bwd_peak_gib"] = (st["peak_live_bytes"] - base) / (1 << 30)
        res["after_live_gib"] = (st["live_bytes"] - base) / (1 << 30)
    except Exception as ex:      # noqa: BLE001
        st = rt.pool_stats()
        res["error"] = "%s: %s" % (type(ex).__name__, str(ex)[:120])
        res["after_live_gib"] = (st["live_bytes"] - base) / (1 << 30)

    print("@@RESULT@@" + json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
