# -*- coding: utf-8 -*-
"""在 `checkpoint` 的每个 block 反向前后打点，看它到底有没有逐块释放。

## 判据

* 每块的 `after` 与 `before` 相近 ⇒ 该块自己占的就这么多，**没跨块累积** ⇒ 问题在别处
* 每块 `after` 比 `before` 高一个固定量、且**单调上升** ⇒ **跨块累积**，块内张量没释放

用法：python ckpt_trace.py [L] [S] [B]
"""
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
OUT = os.path.join(HERE, "ckpttrace.txt")

D, NH, FF = 1280, 10, 5120
DH = D // NH


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 2

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

    base = rt.pool_stats()["live_bytes"]
    trace = []
    orig = T.checkpoint

    def patched(fn, *inputs):
        res = orig(fn, *inputs)
        inner = res._backward
        tag = res

        def bw(g, grads):
            b0 = rt.pool_stats()["live_bytes"]
            inner(g, grads)
            b1 = rt.pool_stats()["live_bytes"]
            trace.append((len(trace), (b0 - base) / (1 << 30), (b1 - base) / (1 << 30)))
        res._backward = bw
        return tag

    T.checkpoint = patched
    x = x0
    for i in range(L):
        x = T.checkpoint(lambda h, _i=i: block(h, _i), x)
    T.checkpoint = orig

    fwd = (rt.pool_stats()["live_bytes"] - base) / (1 << 30)
    loss = (x.reshape(-1, D) * x.reshape(-1, D)).sum()
    loss.backward()
    rt.sync()
    peak = (rt.pool_stats()["peak_live_bytes"] - base) / (1 << 30)
    after = (rt.pool_stats()["live_bytes"] - base) / (1 << 30)

    rep = ["配置 L=%d S=%d B=%d（dh=%d，lowmem 不生效）" % (L, S, B, DH),
           "前向活 %.3f GiB｜反向峰值 %.3f GiB｜结束后 %.3f GiB" % (fwd, peak, after),
           "",
           "| 块序号（反向顺序）| 进入时 GiB | 退出时 GiB | 本块净增 |",
           "| --- | --- | --- | --- |"]
    for i, b0, b1 in trace:
        rep.append("| %d | %.3f | %.3f | %+.3f |" % (i, b0, b1, b1 - b0))
    rep.append("")
    rep.append("判读：**退出时逐块上升 ⇒ 跨块累积**（块内张量没释放）；")
    rep.append("若退出时平稳、只在块内尖峰 ⇒ 单块成本高，不是累积。")
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
