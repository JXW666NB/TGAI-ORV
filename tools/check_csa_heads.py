# -*- coding: utf-8 -*-
"""核对库里 `CausalSelfAttention` 的 head 切分是否正确。

参照实现：显式按 head 切（逐头算），与库实现对照。
若两者不一致，说明库里的 `reshape(B,S,H,Dh).reshape(B*H,S,Dh)` 用错了 ——
从 [B,S,H,Dh] 直接 reshape 到 [B*H,S,Dh] 会把 head 和 seq 混在一起，
正确做法是先 permute 到 [B,H,S,Dh]。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "python"))

import tgai_go.train as T


def reference(x, wq, wk, wv, wo, H, Dh):
    """逐头实现的因果注意力（作为参照）。"""
    B, S, D = x.shape
    q = x.reshape(B * S, D) @ wq.T
    k = x.reshape(B * S, D) @ wk.T
    v = x.reshape(B * S, D) @ wv.T
    q = q.reshape(B, S, H, Dh)
    k = k.reshape(B, S, H, Dh)
    v = v.reshape(B, S, H, Dh)
    outs = []
    for b in range(B):
        for h in range(H):
            qh = q[b, :, h, :]                      # [S, Dh]
            kh = k[b, :, h, :]
            vh = v[b, :, h, :]
            sc = qh @ kh.T / np.sqrt(Dh)
            m = np.tril(np.ones((S, S), np.float32))
            sc = np.where(m > 0, sc, -1e30)
            sc = sc - sc.max(axis=-1, keepdims=True)
            p = np.exp(sc)
            p = p / p.sum(axis=-1, keepdims=True)
            outs.append((b, h, p @ vh))             # [S, Dh]
    o = np.zeros((B, S, H, Dh), np.float32)
    for b, h, val in outs:
        o[b, :, h, :] = val
    o = o.reshape(B, S, H * Dh)
    return (o.reshape(B * S, D) @ wo.T).reshape(B, S, D)


def main():
    T.set_default_device("cpu")
    rng = np.random.default_rng(0)
    B, S, D, H = 2, 5, 8, 2
    Dh = D // H
    x_np = rng.normal(0, 0.5, (B, S, D)).astype(np.float32)
    wq = rng.normal(0, 0.3, (D, D)).astype(np.float32)
    wk = rng.normal(0, 0.3, (D, D)).astype(np.float32)
    wv = rng.normal(0, 0.3, (D, D)).astype(np.float32)
    wo = rng.normal(0, 0.3, (D, D)).astype(np.float32)

    ref = reference(x_np, wq, wk, wv, wo, H, Dh)

    att = T.nn.CausalSelfAttention(D, H)
    for mod, arr in ((att.q_proj, wq), (att.k_proj, wk), (att.v_proj, wv), (att.out_proj, wo)):
        mod.weight = T.nn.Parameter(arr.copy())
    got = np.asarray(att(T.tensor(x_np)).to_numpy(), np.float32)

    d = float(np.abs(got - ref).max())
    scale = max(float(np.abs(ref).max()), 1e-9)
    print("库实现 vs 逐头参照：绝对差 %.3e，相对差 %.2e" % (d, d / scale))
    print("结论：%s" % ("一致" if d / scale <= 1e-5 else "不一致 —— head 切分用错了"))
    return 0 if d / scale <= 1e-5 else 1


if __name__ == "__main__":
    sys.exit(main())
