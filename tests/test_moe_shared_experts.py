# -*- coding: utf-8 -*-
"""共享专家（shared experts）的语义测试。

## 为什么要有这个功能

现代主流 MoE 都带共享专家：DeepSeek-V3 是 **256 路由 + 1 共享**、top-8；
Llama 4 Scout/Maverick 各 **1 个共享**。共享专家**恒定激活、不参与路由**，
作用是吸收通用知识，让路由专家专注特化。

ORV 之前完全不支持（代码里 `shared` 出现 0 次），因此训不了这些结构。

## 语义（DeepSeekMoE）

```
y = Σ_{i∈TopK} g_i · E_i(x)   +   Σ_j S_j(x)
    └── 路由部分（加权） ──┘     └ 共享部分（不加权、全 token）┘
```

注意共享部分是**直接相加、不带门控权重**，也不参与 top-k 竞争。

## 守住什么

1. `n_shared=0` 时**行为与加入该功能前完全一致**（向后兼容）；
2. `n_shared>0` 时输出 == 路由输出 + 共享输出（逐项可验证）；
3. 共享专家**看到全部 token**（与路由无关）；
4. 梯度能流到共享专家；
5. 参数量按 `n_shared × 3 个矩阵` 增加。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import _moe_swiglu


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield
    T.set_default_device("cpu")


def _np(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)


def _fill(mod, seed):
    rng = np.random.default_rng(seed)
    for p in mod.parameters():
        p.data = rng.normal(0, 0.05, tuple(int(s) for s in p.shape)).astype(np.float32)


def test_zero_shared_is_backward_compatible():
    """n_shared=0（默认）必须与旧行为完全一致：同一份权重、同一个输入，输出逐位相同。"""
    D, F, E, K = 16, 32, 4, 2
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (8, D)).astype(np.float32)

    a = T.nn.MoE(D, F, E, top_k=K, gate="softmax")                 # 旧签名
    b = T.nn.MoE(D, F, E, top_k=K, gate="softmax", n_shared=0)     # 显式 0
    _fill(a, 1)
    b.load_state_dict(a.state_dict())

    ya = _np(a(T.tensor(X)))
    yb = _np(b(T.tensor(X)))
    assert np.abs(ya - yb).max() == 0.0, "n_shared=0 不应改变任何数值"


def test_shared_output_is_added_without_gating():
    """输出必须等于「路由输出 + 共享输出」，且共享部分**不带门控权重**。"""
    D, F, E, K = 16, 32, 4, 2
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (8, D)).astype(np.float32)
    Xt = T.tensor(X)

    base = T.nn.MoE(D, F, E, top_k=K, gate="softmax")              # 只有路由
    full = T.nn.MoE(D, F, E, top_k=K, gate="softmax", n_shared=1)  # 路由 + 1 共享
    _fill(base, 2)
    # full 的路由部分与 base 完全相同；共享专家单独填充
    _fill(full, 2)
    rng_s = np.random.default_rng(99)
    for p in full.shared_experts.parameters():
        p.data = rng_s.normal(0, 0.05, tuple(int(s) for s in p.shape)).astype(np.float32)

    y_base = _np(base(Xt))          # 纯路由
    y_full = _np(full(Xt))
    shared = _np(_moe_swiglu(Xt, full.shared_experts[0]))   # 共享专家在全部 token 上的输出

    got = y_full - y_base
    rel = float(np.abs(got - shared).max() / max(1e-9, float(np.abs(shared).max())))
    assert rel <= 1e-6, "共享部分应是『不加权、全 token』的相加，实测相对差 %.2e" % rel


def test_shared_expert_sees_all_tokens():
    """把路由专家输出清零后，输出应**只剩**共享部分 —— 证明共享专家看到全部 token。"""
    D, F, E, K = 8, 16, 2, 1
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (6, D)).astype(np.float32)
    m = T.nn.MoE(D, F, E, top_k=K, gate="softmax", n_shared=1)
    _fill(m, 3)
    # 路由专家的 down 权重清零 ⇒ 路由输出为 0
    for e in m.experts:
        e.down.weight.data = np.zeros(tuple(int(s) for s in e.down.weight.shape), np.float32)

    y = _np(m(T.tensor(X)))
    shared = _np(_moe_swiglu(T.tensor(X), m.shared_experts[0]))
    rel = float(np.abs(y - shared).max() / max(1e-9, float(np.abs(shared).max())))
    assert rel <= 1e-6, "路由清零后应只剩共享输出，实测相对差 %.2e" % rel
    assert np.abs(y).max() > 0, "共享专家没有给全部 token 贡献输出"


def test_gradient_flows_to_shared_experts():
    D, F, E, K = 8, 16, 2, 1
    rng = np.random.default_rng(0)
    X = T.tensor(rng.normal(0, 1, (4, D)).astype(np.float32))
    m = T.nn.MoE(D, F, E, top_k=K, gate="softmax", n_shared=1)
    _fill(m, 4)
    m(T.tensor(np.asarray(X.to_numpy()))).sum().backward()
    g = m.shared_experts[0].down.weight.grad
    assert g is not None, "共享专家没有拿到梯度"
    assert np.abs(np.asarray(g, np.float32)).max() > 0, "共享专家梯度全零"


def test_param_count_grows_by_three_matrices_per_shared_expert():
    D, F, E = 8, 16, 2
    a = T.nn.MoE(D, F, E, n_shared=0)
    b = T.nn.MoE(D, F, E, n_shared=1)
    c = T.nn.MoE(D, F, E, n_shared=2)
    na, nb, nc = (len(list(m.parameters())) for m in (a, b, c))
    per = nb - na
    assert per == 3, "每个共享专家应有 gate/up/down 三个矩阵，实测增加 %d" % per
    assert nc - nb == per, "共享专家数量应线性增加参数"


def test_shared_d_ff_can_differ():
    """共享专家的 d_ff 允许独立设置（有些结构里共享专家更宽）。"""
    D, F, ES = 8, 16, 24
    m = T.nn.MoE(D, F, 2, n_shared=1, shared_d_ff=ES)
    assert m.shared_d_ff == ES
    assert tuple(int(s) for s in m.shared_experts[0].gate.weight.shape) == (ES, D)


def test_rejects_negative_shared():
    with pytest.raises((ValueError, AssertionError)):
        T.nn.MoE(8, 16, 2, n_shared=-1)
