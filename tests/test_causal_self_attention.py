# -*- coding: utf-8 -*-
"""`nn.CausalSelfAttention` 的 head 切分与梯度（0.9.24 修复的真 BUG）。

## 修复的是什么

原实现：

```python
def split(t):
    return t.reshape(B, S, H, Dh).reshape(B * H, S, Dh)   # 错
```

从 `[B,S,H,Dh]` **直接** reshape 到 `[B*H,S,Dh]` 会把 **head 轴与 seq 轴混在一起**，
等于把不同 head、不同位置的数拼错。实测与逐头参照相差 **9.61e-01**（完全算错），
而且**不报错**。正确写法是先 `permute(0,2,1,3)` 把 head 提到 seq 前面再 reshape。

## 这条测试守住什么

1. 前向与 torch 的正确实现一致（≤1e-5）；
2. 反向梯度也与 torch 一致（head 切分错时梯度同样会错，只看前向会漏）；
3. `B>1` 且 `H>1` —— 只有这种形状才暴露该 BUG（B=1 或 H=1 时两个 reshape 恰好等价）。
"""
import numpy as np
import pytest

import tgai_go.train as T

torch = pytest.importorskip("torch")


def _np(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


def _has_cuda():
    try:
        rt = T._cuda()
        return bool(rt is not None and rt.free_device_bytes() is not None)
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _cpu_default():
    T.set_default_device("cpu")
    yield
    T.set_default_device("cpu")


def _build(D, H, seed=0):
    rng = np.random.default_rng(seed)
    ws = [rng.normal(0, 0.25, (D, D)).astype(np.float32) for _ in range(4)]
    att = T.nn.CausalSelfAttention(D, H)
    for mod, arr in zip((att.q_proj, att.k_proj, att.v_proj, att.out_proj), ws):
        mod.weight = T.nn.Parameter(arr.copy())
    return att, ws


def _torch_ref(x_np, ws, D, H, gw=None):
    """torch 的正确实现（显式 permute），返回 (out, grads)。"""
    B, S, _ = x_np.shape
    Dh = D // H
    x = torch.from_numpy(x_np).requires_grad_(True)
    w = [torch.from_numpy(a).requires_grad_(True) for a in ws]
    q = (x.reshape(B * S, D) @ w[0].T).reshape(B, S, H, Dh).permute(0, 2, 1, 3)
    k = (x.reshape(B * S, D) @ w[1].T).reshape(B, S, H, Dh).permute(0, 2, 1, 3)
    v = (x.reshape(B * S, D) @ w[2].T).reshape(B, S, H, Dh).permute(0, 2, 1, 3)
    att = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    o = att.permute(0, 2, 1, 3).reshape(B, S, H * Dh)
    out = (o.reshape(B * S, D) @ w[3].T).reshape(B, S, D)
    if gw is None:
        return out.detach().numpy()
    (out * torch.from_numpy(gw)).sum().backward()
    return out.detach().numpy(), [a.grad.numpy() for a in w]


@pytest.mark.parametrize("B,S,D,H", [(2, 5, 8, 2), (3, 4, 12, 3), (1, 6, 8, 2)])
def test_forward_matches_torch(B, S, D, H):
    """B>1 且 H>1 才暴露原 BUG；B=1 时两个 reshape 等价，作为对照。"""
    rng = np.random.default_rng(B * 10 + H)
    x_np = rng.normal(0, 0.5, (B, S, D)).astype(np.float32)
    att, ws = _build(D, H, seed=B + H)
    got = _np(att(T.tensor(x_np)))
    ref = _torch_ref(x_np, ws, D, H)
    assert _rel(got, ref) <= 1e-5, (B, S, D, H)


@pytest.mark.parametrize("B,S,D,H", [(2, 5, 8, 2), (3, 4, 12, 3)])
def test_backward_matches_torch(B, S, D, H):
    """梯度同样要一致 —— head 切分错时前向错、反向也错，但只测前向会漏掉某些形状。"""
    rng = np.random.default_rng(B * 7 + H)
    x_np = rng.normal(0, 0.5, (B, S, D)).astype(np.float32)
    gw = rng.normal(0, 1, (B, S, D)).astype(np.float32)
    att, ws = _build(D, H, seed=B + H)

    x = T.tensor(x_np, requires_grad=True)
    out = att(x)
    (out * T.tensor(gw)).sum().backward()

    _ref_out, ref_grads = _torch_ref(x_np, ws, D, H, gw=gw)
    ours = [m.weight.grad for m in (att.q_proj, att.k_proj, att.v_proj, att.out_proj)]
    for i, (g, r) in enumerate(zip(ours, ref_grads)):
        assert _rel(_np(g), r) <= 1e-4, "第 %d 个权重梯度不一致" % i


@pytest.mark.skipif(not _has_cuda(), reason="没有可用 CUDA 设备")
def test_cuda_matches_cpu():
    B, S, D, H = 2, 5, 8, 2
    rng = np.random.default_rng(99)
    x_np = rng.normal(0, 0.5, (B, S, D)).astype(np.float32)
    att, _ws = _build(D, H, seed=5)
    ref = _np(att(T.tensor(x_np)))
    T.set_default_device("cuda")
    try:
        got = _np(att(T.tensor(x_np)))
    finally:
        T.set_default_device("cpu")
    assert _rel(got, ref) <= 1e-5
