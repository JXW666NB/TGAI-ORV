# -*- coding: utf-8 -*-
"""量化"梯度累加"的 fp32 噪声底：误差随微批数怎么变。

背景：数学上 4 个微批的梯度之和 == 一次大 batch 的梯度；但 fp32 下跨微批相加有舍入。
目标的判据是 ≤1e-6，实测均值口径 1.30e-06、求和口径 5.68e-06 —— 需要给出"噪声底曲线"，
而不是把判据放宽了事。

运行：python tools/accum_floor.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python") if False else None
sys.path.insert(0, r"python")

import tgai_go.train as T          # noqa: E402
import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402

D, C, N = 8, 5, 24
rng = np.random.default_rng(0)
X = rng.normal(0, 1, (N, D)).astype(np.float32)
Y = rng.integers(0, C, N).astype(np.int64)


def rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def orv_grads(micro, average=True, w0=None):
    lin = T.nn.Linear(D, C)
    W = lin.weight
    if w0 is not None:
        np.copyto(np.asarray(W.data, np.float32), w0)   # 必须用同一份初始权重，否则比的是不同模型
    bs = N // micro
    acc = T.optim.Accumulator([W], steps=micro, average=average)
    for k in range(micro):
        sl = slice(k * bs, (k + 1) * bs)
        lg = T.tensor(np.ascontiguousarray(X[sl])) @ W.T
        loss = T.nn.functional.cross_entropy(lg, np.ascontiguousarray(Y[sl]))
        loss.backward()
        acc.tick()
    acc.finalize()
    return np.asarray(W.grad.data, np.float32).copy(), np.array(W.data, np.float32, copy=True)


def torch_grad(W_eff):
    p = torch.nn.Parameter(torch.from_numpy(W_eff))
    lg = torch.from_numpy(X) @ p.T
    F.cross_entropy(lg, torch.from_numpy(Y)).backward()
    return p.grad.detach().numpy()


print("N=%d tokens，D=%d，C=%d；每行：微批数 → 梯度相对误差" % (N, D, C))
for micro in (1, 2, 3, 4, 6, 8, 12, 24):
    if N % micro:
        continue
    g, W_eff = orv_grads(micro, average=True)
    ref = torch_grad(W_eff)
    print("  micro=%-3d  mean 口径 相对误差=%.2e" % (micro, rel(g, ref)))

print()
print("★ 隔离'累加本身'：ORV 多微批 vs **ORV 单批**（同一实现，排除跨库 fp32 差异）")
w0 = np.random.default_rng(7).normal(0, 0.2, (C, D)).astype(np.float32)
base = None
for micro in (1, 2, 4, 6, 8):
    if N % micro:
        continue
    g, _ = orv_grads(micro, average=True, w0=w0)
    if micro == 1:
        base = g
        print("  micro=1（基准，无累加）")
        continue
    print("  micro=%-3d  vs ORV 单批 相对误差=%.2e" % (micro, rel(g, base)))

print()
print("★ 同口径下 torch 单批 vs ORV 单批（这就是上一段 2e-6 的来源）")
g1, W_eff = orv_grads(1, average=True, w0=w0)
print("  ORV 单批 vs torch 单批 相对误差=%.2e" % rel(g1, torch_grad(W_eff)))
