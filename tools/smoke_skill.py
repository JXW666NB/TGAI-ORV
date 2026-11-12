# -*- coding: utf-8 -*-
"""skill 里那段最小验证脚本的**逐字副本**，用来实跑确认它没错。

与 skill 的 §6 第 2 步保持一致；改动 skill 里那段代码时，这里必须同步。
"""
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import functional as Fn

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)
V, D, L, NH, FF, S, B = 512, 256, 2, 4, 1024, 128, 4
DH = D // NH


def dev(n, m):
    return T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.02, (n, m)).astype(np.float32)))


qkv = [dev(3 * D, D) for _ in range(L)]
o = [dev(D, D) for _ in range(L)]
g = [dev(FF, D) for _ in range(L)]
u = [dev(FF, D) for _ in range(L)]
dn = [dev(D, FF) for _ in range(L)]
head = dev(V, D)
emb = T.nn.Embedding(V, D)
params = [emb.weight, head] + qkv + o + g + u + dn
for t in params:
    t.requires_grad = True

ids = rng.integers(0, V, (B, S)).astype(np.int64)
tgt = rng.integers(0, V, (B, S)).astype(np.int64)
x = emb(ids.reshape(-1)).reshape(B, S, D)
for i in range(L):
    z = x.reshape(-1, D) @ qkv[i].T
    q, k, v = z.reshape(B, S, 3, NH, DH).permute(2, 0, 3, 1, 4)
    y = Fn.scaled_dot_product_attention(q.reshape(B * NH, S, DH),
                                        k.reshape(B * NH, S, DH),
                                        v.reshape(B * NH, S, DH), is_causal=True)
    x = x + (y.reshape(B, NH, S, DH).permute(0, 2, 1, 3)
             .reshape(-1, D) @ o[i].T).reshape(B, S, D)
    f = x.reshape(-1, D)
    x = x + ((Fn.silu(f @ g[i].T) * (f @ u[i].T)) @ dn[i].T).reshape(B, S, D)

opt = T.optim.AdamW(params, lr=1e-3)
first = last = None
for step in range(20):
    loss = Fn.cross_entropy((x.reshape(-1, D) @ head.T).reshape(-1, V), tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    rt.sync()
    val = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
    first = val if first is None else first
    last = val
    if step in (0, 19):
        st = rt.pool_stats()
        print("step %2d loss %.6f 峰值 %.3f GiB"
              % (step, val, st["peak_live_bytes"] / (1 << 30)))
print("RESULT first=%.6f last=%.6f 下降=%s" % (first, last, last < first))
