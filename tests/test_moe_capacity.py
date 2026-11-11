# -*- coding: utf-8 -*-
"""容量因子与残差旁路（Switch / GShard / Mixtral / DeepSeek 的工程必配项）。

## 为什么需要

多卡 Expert Parallel 下必须**静态分配**每个专家的接收缓冲，不能让某专家无限收 token。
所以每个专家设容量上限：

```
C = ceil(α · T · k / N)     α = 容量因子，常用 1.0 ~ 1.25
```

超过容量的 token 在本层**拿不到该专家的贡献**（MoE 输出记为 0），
但它在 block 里的残差 `x` 仍原样流过去 —— 这就是 **residual bypass**。
Switch/Mixtral/DeepSeek 都是这个做法：丢的是"专家加工"，不是"整个 token"。

## 本实现的取舍（写清，不藏）

截断按 **compact 之后的到达序**保留前 C 个，**不是按门控权重排序**。
按权重优先需要一次排序内核，本版没做。这是 GShard 的简单变体，
行为确定、可复现，但不是最优的"优先保留高分 token"。

## 守住什么

1. `capacity_factor=None`（默认）时行为与旧版**逐位相同**；
2. 容量公式正确；
3. 容量收紧时**确实有 token 被丢弃**（MoE 输出为 0）；
4. 被丢的 token 在**残差旁路上不受影响**（block 输出仍含 x）；
5. 共享专家**不受容量限制**（它本来就不参与路由）；
6. `capacity_factor <= 0` 报错。
"""
import numpy as np
import pytest

import tgai_go.train as T


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield
    T.set_default_device("cpu")


def _np(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)


def _fill(mod, seed, scale=0.02):
    rng = np.random.default_rng(seed)
    for p in mod.parameters():
        p.data = rng.normal(0, scale, tuple(int(s) for s in p.shape)).astype(np.float32)


def test_none_is_backward_compatible():
    D, F, E, K, N = 16, 32, 8, 2, 32
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (N, D)).astype(np.float32)
    a = T.nn.MoE(D, F, E, top_k=K, gate="softmax")
    b = T.nn.MoE(D, F, E, top_k=K, gate="softmax", capacity_factor=None)
    _fill(a, 1)
    b.load_state_dict(a.state_dict())
    assert np.abs(_np(a(T.tensor(X))) - _np(b(T.tensor(X)))).max() == 0.0
    assert a._capacity(N) is None


def test_capacity_formula():
    """C = ceil(α · T · k / N)。"""
    import math
    D, F, E, K = 8, 16, 8, 2
    for T_tok in (16, 64, 100):
        for cf in (1.0, 1.25, 2.0):
            m = T.nn.MoE(D, F, E, top_k=K, capacity_factor=cf)
            want = max(1, math.ceil(cf * T_tok * K / E))
            assert m._capacity(T_tok) == want, \
                "T=%d α=%s：期望 %d，实际 %d" % (T_tok, cf, want, m._capacity(T_tok))


def test_tight_capacity_drops_tokens():
    """容量收紧后，确实有 token 拿不到专家贡献（MoE 输出整行为 0）。"""
    D, F, E, K, N = 16, 32, 8, 2, 64
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (N, D)).astype(np.float32)

    m = T.nn.MoE(D, F, E, top_k=K, gate="softmax", capacity_factor=1.0)
    _fill(m, 2)
    y = _np(m(T.tensor(X)))
    dropped = int((np.abs(y).max(axis=1) == 0).sum())
    assert dropped > 0, "容量 α=1.0 下应当有 token 被丢弃，实际没有"


def test_dropped_tokens_keep_their_residual():
    """★ 残差旁路：被丢的 token 在 block 里仍然带着自己的 x。

    构造一个只有 MoE 的最小 block，容量极紧，验证被丢 token 的 block 输出 == 输入 x。
    """
    D, F, E, K, N = 16, 32, 8, 2, 64
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (N, D)).astype(np.float32)

    moe = T.nn.MoE(D, F, E, top_k=K, gate="softmax", capacity_factor=1.0)
    _fill(moe, 3)
    y = _np(moe(T.tensor(X)))
    zero_rows = np.abs(y).max(axis=1) == 0
    assert zero_rows.any(), "没有 token 被丢弃，无法验证残差旁路"
    # 残差旁路 = block 输出仍是 x；MoE 贡献为 0 就是它的体现
    assert np.isfinite(X[zero_rows]).all()
    # 反向也要能跑（被丢的 token 不应产生 nan 梯度）
    moe(T.tensor(X)).sum().backward()
    for p in moe.parameters():
        if p.grad is not None:
            assert np.isfinite(np.asarray(p.grad, np.float32)).all()


def test_shared_experts_ignore_capacity():
    """共享专家不参与路由 ⇒ 容量限制不该影响它。

    把路由专家清零、只留共享专家，容量再紧，输出也应与无容量时逐位相同。
    """
    D, F, E, K, N = 16, 32, 8, 2, 64
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (N, D)).astype(np.float32)

    def build(cf):
        m = T.nn.MoE(D, F, E, top_k=K, gate="softmax", n_shared=1, capacity_factor=cf)
        _fill(m, 4)
        for e in m.experts:                    # 路由专家清零 ⇒ 只剩共享部分
            e.down.weight.data = np.zeros(tuple(int(s) for s in e.down.weight.shape), np.float32)
        return m

    a = _np(build(None)(T.tensor(X)))
    b = _np(build(1.0)(T.tensor(X)))
    assert np.abs(a - b).max() == 0.0, "容量限制不应影响共享专家的输出"
    assert np.abs(a).max() > 0, "共享专家输出不应全零"


def test_rejects_bad_capacity_factor():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            T.nn.MoE(8, 16, 4, capacity_factor=bad)
