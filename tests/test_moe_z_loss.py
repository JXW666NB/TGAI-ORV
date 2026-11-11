# -*- coding: utf-8 -*-
"""Z-loss（ST-MoE, arXiv:2202.08906）—— MoE router 的数值稳定手段。

## 它解决什么问题

fp16 下 router logits 容易溢出（早期训练 router 权重摆动大），溢出会让 softmax
退化成 one-hot 甚至 nan。Z-loss 惩罚 logit 的绝对大小：

```
L_z = β · Σ_t (logsumexp(s_t))²
```

## 实现技巧

`log_softmax(x)_j = x_j − lse`，所以 `x_j − log_softmax_j === lse` **对每一列都成立**。
取均值就得到 lse，不需要新写 logsumexp 内核。

## 守住什么

1. `z_loss_coef=0`（默认）时不算、`self.z_loss is None`，行为与旧版一致；
2. 数值与手算 `β·Σ(logsumexp(s))²` 一致（≤1e-6 相对差）；
3. logit 幅值变大时 z_loss 单调变大（这才是它"压制幅值"的作用）；
4. 梯度能流到 router；
5. **它不影响前向输出**（只是额外返回一个 loss 项）。
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


def _manual(logits, coef):
    lg = np.asarray(logits, np.float64)
    m = lg.max(-1, keepdims=True)
    lse = (m + np.log(np.exp(lg - m).sum(-1, keepdims=True))).reshape(-1)
    return coef * float((lse ** 2).sum())


def test_disabled_by_default():
    m = T.nn.MoE(16, 32, 8, top_k=2)
    _fill(m, 1)
    m(T.tensor(np.random.default_rng(0).normal(0, 1, (16, 16)).astype(np.float32)))
    assert m.z_loss is None, "默认不应计算 z_loss"


def test_matches_manual_formula():
    D, F, E, K, N = 16, 32, 8, 2, 24
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (N, D)).astype(np.float32)
    coef = 1e-3
    m = T.nn.MoE(D, F, E, top_k=K, gate="softmax", z_loss_coef=coef)
    _fill(m, 2)
    m(T.tensor(X))
    logits = np.asarray((T.tensor(X) @ m.router.weight.T).data, np.float32)
    want = _manual(logits, coef)
    got = float(np.asarray(m.z_loss.data))
    rel = abs(got - want) / max(1e-9, abs(want))
    assert rel <= 1e-6, "z_loss 与手算不一致：实现 %.8f vs 手算 %.8f" % (got, want)


def test_grows_with_logit_magnitude():
    """router 权重放大 ⇒ logits 放大 ⇒ z_loss 必须变大（这就是"压制幅值"的作用）。"""
    D, F, E, N = 16, 32, 8, 16
    rng = np.random.default_rng(0)
    X = T.tensor(rng.normal(0, 1, (N, D)).astype(np.float32))

    vals = []
    for scale in (0.02, 0.2, 2.0):
        m = T.nn.MoE(D, F, E, top_k=2, z_loss_coef=1e-3)
        _fill(m, 3, scale=scale)
        m(X)
        vals.append(float(np.asarray(m.z_loss.data)))
    assert vals[0] < vals[1] < vals[2], "z_loss 未随 logit 幅值单调增长：%s" % vals


def test_gradient_flows_to_router():
    D, F, E, N = 8, 16, 4, 8
    rng = np.random.default_rng(0)
    X = T.tensor(rng.normal(0, 1, (N, D)).astype(np.float32))
    m = T.nn.MoE(D, F, E, top_k=2, z_loss_coef=1e-2)
    _fill(m, 4)
    m(X)
    m.z_loss.backward()
    g = m.router.weight.grad
    assert g is not None, "z_loss 没有把梯度传到 router"
    assert np.abs(np.asarray(g, np.float32)).max() > 0, "router 梯度全零"


def test_does_not_change_forward_output():
    """开不开 z_loss，前向输出必须逐位相同 —— 它只是额外返回一个 loss 项。"""
    D, F, E, N = 16, 32, 8, 16
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (N, D)).astype(np.float32)
    a = T.nn.MoE(D, F, E, top_k=2, gate="softmax")
    b = T.nn.MoE(D, F, E, top_k=2, gate="softmax", z_loss_coef=1e-3)
    _fill(a, 5)
    b.load_state_dict(a.state_dict())
    ya = _np(a(T.tensor(X)))
    yb = _np(b(T.tensor(X)))
    assert np.abs(ya - yb).max() == 0.0, "z_loss 不应改变前向输出"
    assert b.z_loss is not None


def test_composes_with_aux_free_and_shared():
    """三个新特性同开也要正常：共享专家 + aux_free + z_loss。"""
    D, F, E, N = 16, 32, 8, 16
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (N, D)).astype(np.float32)
    m = T.nn.MoE(D, F, E, top_k=2, gate="softmax",
                 n_shared=1, balance="aux_free", z_loss_coef=1e-3)
    _fill(m, 6)
    y = _np(m(T.tensor(X)))
    assert np.isfinite(y).all()
    assert float(np.abs(y).max()) > 0
    assert m.z_loss is not None and float(np.asarray(m.z_loss.data)) > 0
    assert float(np.abs(m.expert_bias).max()) > 0, "aux_free 的 bias 未被更新"
