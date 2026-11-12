# -*- coding: utf-8 -*-
"""检查点不释放的根因：是引用环，还是有人持有强引用？

## 判据

逐层跑前向（检查点开），每层后打印：
  live_before_gc / live_after_gc

* **gc.collect() 之后明显下降** ⇒ 是**引用环**（`_backward` 闭包 ↔ 张量、
  或 parent ↔ child 互指）。修法：切断环，或改用 weakref。
* **gc 之后不降** ⇒ 有**强引用**还在（池的登记表、keepalive 链、闭包捕获）。
  修法：找到持有者。

同时统计 Tensor / GpuBuf 的存活对象数，看它们是否随层数累积。
"""
import gc
import io
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
OUT = os.path.join(HERE, "ckptdiag.txt")

import sys as _s
D, NH, FF, S, B = 1280, 10, 5120, int(_s.argv[1]) if len(_s.argv)>1 else 1024, int(_s.argv[2]) if len(_s.argv)>2 else 1
DH = D // NH


def main():
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    rep = []

    def dev(n, m):
        return T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.02, (n, m)).astype(np.float32)))

    L = 6
    qkv = [dev(3 * D, D) for _ in range(L)]
    o = [dev(D, D) for _ in range(L)]
    gate = [dev(FF, D) for _ in range(L)]
    up = [dev(FF, D) for _ in range(L)]
    down = [dev(D, FF) for _ in range(L)]
    for t in qkv + o + gate + up + down:
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

    def count(kinds):
        n = {k: 0 for k in kinds}
        for ob in gc.get_objects():
            tn = type(ob).__name__
            if tn in n:
                n[tn] += 1
        return n

    base = rt.pool_stats()["live_bytes"]
    rep.append("配置：L=%d D=%d dh=%d FF=%d S=%d B=%d" % (L, D, DH, FF, S, B))
    rep.append("")
    rep.append("| 层 | 活着(GiB) | gc 后(GiB) | 差 | Tensor 数 | GpuBuf 数 |")
    rep.append("| --- | --- | --- | --- | --- | --- |")

    x = x0
    for i in range(L):
        x = T.checkpoint(lambda h, _i=i: block(h, _i), x)
        live1 = (rt.pool_stats()["live_bytes"] - base) / (1 << 30)
        c1 = count(("Tensor", "GpuBuf"))
        gc.collect()
        live2 = (rt.pool_stats()["live_bytes"] - base) / (1 << 30)
        c2 = count(("Tensor", "GpuBuf"))
        rep.append("| %d | %.3f | %.3f | %.3f | %d→%d | %d→%d |"
                   % (i + 1, live1, live2, live1 - live2,
                      c1["Tensor"], c2["Tensor"], c1["GpuBuf"], c2["GpuBuf"]))

    # ---- 反向阶段：p100_fit 是在这里 OOM 的 ----
    live_fwd = (rt.pool_stats()["live_bytes"] - base) / (1 << 30)
    rep.append("")
    rep.append("前向结束后活张量 %.3f GiB" % live_fwd)
    try:
        loss = (x.reshape(-1, D) * x.reshape(-1, D)).sum()
        loss.backward()
        rt.sync()
        live_bwd = (rt.pool_stats()["live_bytes"] - base) / (1 << 30)
        peak_bwd = (rt.pool_stats()["peak_live_bytes"] - base) / (1 << 30)
        rep.append("反向结束后活 %.3f GiB；**反向峰值 %.3f GiB**"
                   % (live_bwd, peak_bwd))
        gc.collect()
        rep.append("gc 后活 %.3f GiB"
                   % ((rt.pool_stats()["live_bytes"] - base) / (1 << 30)))
    except Exception as ex:      # noqa: BLE001
        st = rt.pool_stats()
        rep.append("反向失败：%s；当时活 %.3f GiB"
                   % (type(ex).__name__, (st["live_bytes"] - base) / (1 << 30)))
    rep.append("")
    rep.append("判读：**gc 后明显下降 ⇒ 引用环**；不降 ⇒ 强引用还在。")
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
