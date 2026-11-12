# -*- coding: utf-8 -*-
"""最小训练冒烟：只依赖 tgai_go.train + numpy，不依赖仓库。

这是要给外部新人的脚本，所以必须自带、可独立运行、打印三样东西：
loss 首末、峰值活张量、每步耗时。
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


emb_w = dev(V, D)
qkv = [dev(3 * D, D) for _ in range(L)]
o = [dev(D, D) for _ in range(L)]
g = [dev(FF, D) for _ in range(L)]
u = [dev(FF, D) for _ in range(L)]
dn = [dev(D, FF) for _ in range(L)]
head = dev(V, D)
for t in [emb_w, head] + qkv + o + g + u + dn:
    t.requires_grad = True

ids = rng.integers(0, V, (B, S)).astype(np.int64)
tgt = rng.integers(0, V, (B, S)).astype(np.int64)

emb_layer = T.nn.Embedding(V, D, weight=T.nn.Parameter(np.zeros((V, D), np.float32)))
emb_layer.weight._gpu = emb_w._gpu
x = emb_layer(ids.reshape(-1)).reshape(B, S, D)

for i in range(L):
    z = x.reshape(-1, D) @ qkv[i].T
    q, k, v = z.reshape(B, S, 3, NH, DH).permute(2, 0, 3, 1, 4)
    y = Fn.scaled_dot_product_attention(q.reshape(B * NH, S, DH),
                                        k.reshape(B * NH, S, DH),
                                        v.reshape(B * NH, S, DH), is_causal=True)
    x = x + (y.reshape(B, NH, S, DH).permute(0, 2, 1, 3)
             .reshape(-1, D) @ o[i].T).reshape(B, S, D)
    f = x.reshape(-1, D)
    h = Fn.silu(f @ g[i].T) * (f @ u[i].T)
    x = x + (h @ dn[i].T).reshape(B, S, D)

params = [emb_w, head] + qkv + o + g + u + dn
opt = T.optim.AdamW(params, lr=1e-3)

first = None
last = None
for step in range(20):
    t0 = time.perf_counter()
    logits = (x.reshape(-1, D) @ head.T).reshape(-1, V)
    loss = Fn.cross_entropy(logits, tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    rt.sync()
    val = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
    if step == 0:
        first = val
    last = val
    if step in (0, 19):
        st = rt.pool_stats()
        print("step %2d  loss %.6f  峰值活张量 %.3f GiB  耗时 %.1f ms"
              % (step, val, st["peak_live_bytes"] / (1 << 30),
                 (time.perf_counter() - t0) * 1e3))

print()
print("RESULT loss_first=%.6f loss_last=%.6f 下降=%s"
      % (first, last, "是" if last < first else "否"))
