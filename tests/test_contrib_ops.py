# -*- coding: utf-8 -*-
"""contrib 官方插件的 PyTorch 对照测试（纯 CPU）。

约定：每个 contrib 算子都必须在**前向与反向**上与 torch 一致（不可导的只验前向 + 梯度为 0）。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train.contrib import autoload

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    T.set_default_device("cpu")
    autoload()


def _t(a):
    return T.tensor(np.ascontiguousarray(np.asarray(a, dtype=np.float32)),
                    requires_grad=True)


# (名字, torch 参考, 输入, kwargs)
DIFF_CASES = [
    ("erf", lambda t: torch.erf(t), [-1.5, 0.0, 2.0, 0.7], {}),
    ("gelu_exact", lambda t: torch.nn.functional.gelu(t), [-1.5, 0.0, 2.0, 0.7], {}),
    ("elu", lambda t: torch.nn.functional.elu(t), [-1.5, 0.0, 2.0, 0.7], {}),
    ("selu", lambda t: torch.nn.functional.selu(t), [-1.5, 0.0, 2.0, 0.7], {}),
    ("leaky_relu", lambda t: torch.nn.functional.leaky_relu(t), [-1.5, 0.0, 2.0, 0.7], {}),
    ("softplus", lambda t: torch.nn.functional.softplus(t), [-1.5, 0.0, 2.0, 0.7], {}),
    ("mish", lambda t: torch.nn.functional.mish(t), [-1.5, 0.0, 2.0, 0.7], {}),
    ("hardswish", lambda t: torch.nn.functional.hardswish(t), [-4.0, -1.5, 0.0, 2.0, 5.0], {}),
    ("hardsigmoid", lambda t: torch.nn.functional.hardsigmoid(t), [-4.0, -1.5, 0.0, 2.0, 5.0], {}),
    # log1p 定义域 x > -1
    ("log1p", lambda t: torch.log1p(t), [-0.9, 0.0, 2.0, 7.5], {}),
]


@pytest.mark.parametrize("name,ref,xs,kw", DIFF_CASES)
def test_differentiable_ops_match_torch(name, ref, xs, kw):
    x_np = np.array([xs], dtype=np.float32)
    x = _t(x_np)
    y = getattr(T.functional, name)(x, **kw) if kw else getattr(T.functional, name)(x)
    y.sum().backward()

    tx = torch.from_numpy(x_np.copy()).requires_grad_(True)
    ty = ref(tx)
    ty.sum().backward()

    assert np.allclose(y.to_numpy(), ty.detach().numpy(), rtol=1e-5, atol=1e-6), name
    assert np.allclose(x.grad, tx.grad.numpy(), rtol=1e-5, atol=1e-6), name


@pytest.mark.parametrize("name,ref", [
    ("sign", lambda t: torch.sign(t)),
    ("floor", lambda t: torch.floor(t)),
    ("ceil", lambda t: torch.ceil(t)),
    ("trunc", lambda t: torch.trunc(t)),
])
def test_zero_grad_ops(name, ref):
    xs = np.array([[-1.7, 0.0, 2.3, 0.5]], dtype=np.float32)
    x = _t(xs)
    y = getattr(T.functional, name)(x)
    y.sum().backward()
    tx = torch.from_numpy(xs.copy())
    assert np.allclose(y.to_numpy(), ref(tx).numpy(), rtol=0, atol=0), name
    assert np.allclose(x.grad, 0.0), name


def test_lerp_multi_input():
    a = _t(np.array([[0.0, 1.0, -2.0]], np.float32))
    b = _t(np.array([[2.0, 3.0, -4.0]], np.float32))
    w = _t(np.array([[0.25, 0.5, 0.75]], np.float32))
    y = T.functional.lerp(a, b, w)
    y.sum().backward()

    ta = torch.tensor([[0.0, 1.0, -2.0]], requires_grad=True)
    tb = torch.tensor([[2.0, 3.0, -4.0]], requires_grad=True)
    tw = torch.tensor([[0.25, 0.5, 0.75]], requires_grad=True)
    torch.lerp(ta, tb, tw).sum().backward()

    assert np.allclose(y.to_numpy(), torch.lerp(ta, tb, tw).detach().numpy(), atol=1e-6)
    for ours, ref in ((a, ta), (b, tb), (w, tw)):
        assert np.allclose(ours.grad, ref.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------- 形状类
def test_flip_forward_and_backward():
    xs = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = _t(xs)
    y = T.functional.flip(x, [1])
    y.sum().backward()
    assert np.allclose(y.to_numpy(), torch.flip(torch.from_numpy(xs), [1]).numpy())
    assert np.allclose(x.grad, np.ones_like(xs))       # flip 的梯度仍是 1


def test_roll_forward_and_backward():
    xs = np.arange(6, dtype=np.float32)
    x = _t(xs)
    y = T.functional.roll(x, [2], [0])
    y.sum().backward()
    assert np.allclose(y.to_numpy(), torch.roll(torch.from_numpy(xs), [2], [0]).numpy())
    assert np.allclose(x.grad, np.ones_like(xs))


def test_flatten_unsqueeze_squeeze():
    xs = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    x = _t(xs)
    y = T.functional.flatten(x, 1, -1)
    y.sum().backward()
    assert y.shape == (2, 12)
    assert np.allclose(y.to_numpy(), torch.flatten(torch.from_numpy(xs), 1, -1).numpy())
    assert np.allclose(x.grad, np.ones_like(xs))

    x2 = _t(xs)
    y2 = T.functional.unsqueeze(x2, 1)
    y2.sum().backward()
    assert y2.shape == (2, 1, 3, 4)
    assert np.allclose(x2.grad, np.ones_like(xs))

    x3 = _t(xs)
    y3 = T.functional.squeeze(x3, 0)      # 第 0 维长度 2，不是 1 ⇒ 形状不消失
    assert y3.shape == (2, 3, 4)


def test_flip_dim_error_names_fix():
    x = _t(np.zeros((2, 3), np.float32))
    with pytest.raises(Exception) as ei:
        T.functional.flatten(x, 0, 5)
    assert "维度" in str(ei.value)


def test_contrib_list_nonempty():
    from tgai_go.train.contrib import list_ops
    names = list_ops()
    assert "erf" in names and "mish" in names and "flip" in names
    assert len(names) >= 18
