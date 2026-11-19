# -*- coding: utf-8 -*-
"""Gradient equivalence: current dispatch vs padded-dense (pad=1.25), real scale.

Both paths get the *same* upstream gradient (loss = sum(out * R)), so dx / dW are
directly comparable.  Routing distribution = real step-18 layer 0.
"""
import math

import numpy as np

import tgai_go.train as T
from tgai_go.train import indexing as IDX
from tgai_go.train import v3_moe as MOE
from tgai_go.train import v3_ops as V
from tgai_go.train import v3_scatter as S

N, K, D, DFF, E = 2048, 2, 1024, 3328, 4
PAD = 1.25
C = int(math.ceil(N * K / float(E) * PAD))

T.set_default_device("cuda")
T.amp_on(dtype="fp16", weight_cache=True)
rt = T._cuda()

rng = np.random.default_rng(7)
xh = rng.normal(0, .3, (N, D)).astype(np.float32)
gh = rng.uniform(.1, 1., (N, K)).astype(np.float32)
rh = rng.normal(0, 1., (N, D)).astype(np.float32)
with np.load(r"D:\orv_bench\moe_route_eids.npz") as z:
    eids = np.asarray(z["s18_l0"], np.int64)

Wn = [[rng.normal(0, .02, (DFF, D)).astype(np.float32) for _ in range(E)],
      [rng.normal(0, .02, (DFF, D)).astype(np.float32) for _ in range(E)],
      [rng.normal(0, .02, (D, DFF)).astype(np.float32) for _ in range(E)]]


def make():
    W = [[T.tensor(w) for w in wl] for wl in Wn]
    for wl in W:
        for w in wl:
            w.requires_grad = True
    x = T.tensor(xh)
    x.requires_grad = True
    return x, W


R = T.tensor(rh)
gates = T.tensor(gh)

# ---------------- path A ----------------
x, W = make()
rt.f16_cache_pin([int(w._gpu.ptr) for wl in W for w in wl])
outA = MOE.moe_topk_orv(x, eids, gates, W[0], W[1], W[2])
(outA * R).sum().backward()
dxA = np.asarray(x.grad.to_numpy() if hasattr(x.grad, "to_numpy") else x.grad, np.float32)
dWA = [[np.asarray(w.grad.to_numpy() if hasattr(w.grad, "to_numpy") else w.grad, np.float32)
        for w in wl] for wl in W]
outA_np = np.asarray(outA.to_numpy(), np.float32)

# ---------------- path B (padded dense) ----------------
x2, W2 = make()
rt.f16_cache_pin([int(w._gpu.ptr) for wl in W2 for w in wl])
flat = eids.reshape(-1)
gw_t, plans = [], []
for e in range(E):
    pos = np.flatnonzero(flat == e)
    arr = np.zeros(C, np.int32)
    used = pos[:C]
    arr[:used.size] = used
    plans.append((arr // K).astype(np.int32))
    gw = np.zeros(C, np.float32)
    gw[:used.size] = gh.reshape(-1)[used]
    gw_t.append(T.tensor(gw))
acc = None
for e in range(E):
    rows = IDX.index_select(x2, 0, plans[e])
    y = V.situ_glu_orv(rows, W2[0][e], W2[1][e], W2[2][e])
    y = y * gw_t[e].reshape(-1, 1)
    z = S.scatter_add(y, plans[e], N)
    acc = z if acc is None else acc + z
(acc * R).sum().backward()
dxB = np.asarray(x2.grad.to_numpy() if hasattr(x2.grad, "to_numpy") else x2.grad, np.float32)
dWB = [[np.asarray(w.grad.to_numpy() if hasattr(w.grad, "to_numpy") else w.grad, np.float32)
        for w in wl] for wl in W2]
outB_np = np.asarray(acc.to_numpy(), np.float32)


def rep(name, a, b):
    d = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    print("%-10s max|d|=%.3e  max_rel=%.3e  mean|d|=%.3e  |a|max=%.3e"
          % (name, d.max(), (d / np.maximum(denom, 1e-12)).max(), d.mean(),
             np.abs(a).max()))


print("[grad] C=%d pad=%.2f  counts=%s" %
      (C, PAD, [int((flat == e).sum()) for e in range(E)]))
rep("out", outA_np, outB_np)
rep("dx", dxA, dxB)
for li, nm in ((0, "dW1"), (1, "dW2"), (2, "dW3")):
    for e in range(E):
        rep("%s[e%d]" % (nm, e), dWA[li][e], dWB[li][e])
