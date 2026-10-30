# -*- coding: utf-8 -*-
"""算子插件系统的测试（纯 CPU，torch 作数值基准）。

覆盖：组合式插件 / 自带反向插件 / 重名保护 / 核心覆盖需显式声明 / 从文件加载 / 卸载。
"""
import os

import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import ops_plugin

torch = pytest.importorskip("torch")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


@pytest.fixture(autouse=True)
def _clean_registry():
    """每个用例前后清理，避免相互污染。"""
    before = set(ops_plugin.registered_names())
    yield
    for name in set(ops_plugin.registered_names()) - before:
        try:
            ops_plugin.unregister(name)
        except Exception:  # noqa: BLE001
            pass


def _t(a):
    return T.tensor(np.ascontiguousarray(np.asarray(a, dtype=np.float32)),
                    requires_grad=True)


def test_composition_plugin_forward_and_backward():
    def sq_plus_one(x):
        return x * x + 1.0

    ops_plugin.register("sq_plus_one", sq_plus_one, kind="tensor")
    assert "sq_plus_one" in ops_plugin.registered_names()

    x_np = np.array([[-2.0, 0.5, 3.0]], np.float32)
    x = _t(x_np)
    y = x.sq_plus_one()
    y.sum().backward()

    tx = torch.from_numpy(x_np.copy()).requires_grad_(True)
    ty = tx * tx + 1.0
    ty.sum().backward()

    assert np.allclose(y.to_numpy(), ty.detach().numpy(), rtol=0, atol=1e-6)
    assert np.allclose(x.grad, tx.grad.numpy(), rtol=0, atol=1e-6)


def test_numpy_plugin_with_custom_backward():
    def fwd(x):
        return np.tanh(x).astype(np.float32)

    def bwd(g, out, xs):
        (x,) = xs
        return [g * (1.0 - np.tanh(x) ** 2)]

    ops_plugin.register_numpy("tanh_custom", fwd, bwd, kind="tensor")
    x_np = np.array([[0.3, -1.2, 2.0]], np.float32)
    x = _t(x_np)
    y = x.tanh_custom()
    y.sum().backward()

    tx = torch.from_numpy(x_np.copy()).requires_grad_(True)
    ty = torch.tanh(tx)
    ty.sum().backward()
    assert np.allclose(y.to_numpy(), ty.detach().numpy(), rtol=0, atol=1e-6)
    assert np.allclose(x.grad, tx.grad.numpy(), rtol=0, atol=1e-6)


def test_duplicate_name_is_rejected():
    ops_plugin.register("dup_op", lambda x: x * 2.0)
    with pytest.raises(ValueError) as ei:
        ops_plugin.register("dup_op", lambda x: x * 3.0)
    msg = str(ei.value)
    assert "已注册" in msg and "修法" in msg, msg


def test_core_name_requires_explicit_replace():
    """不许静默覆盖核心算子（比如 softmax）。"""
    with pytest.raises(ValueError) as ei:
        ops_plugin.register("softmax", lambda x: x)
    assert "核心" in str(ei.value) and "replace=True" in str(ei.value)


def test_bad_name_is_rejected():
    with pytest.raises(ValueError) as ei:
        ops_plugin.register("2bad-name", lambda x: x)
    assert "标识符" in str(ei.value)


def test_numpy_plugin_requires_tensor_input():
    ops_plugin.register_numpy("only_np", lambda x: x, lambda g, o, xs: [g])
    with pytest.raises(TypeError) as ei:
        T.functional.only_np(np.zeros((2,), np.float32))
    assert "Tensor" in str(ei.value)


def test_unregister_removes_from_functional():
    ops_plugin.register("temp_op", lambda x: x + 1.0)
    assert hasattr(T.functional, "temp_op")
    ops_plugin.unregister("temp_op")
    assert not hasattr(T.functional, "temp_op")
    assert "temp_op" not in ops_plugin.registered_names()


def test_unregister_unknown_raises():
    with pytest.raises(KeyError) as ei:
        ops_plugin.unregister("never_registered_op")
    assert "没有注册过" in str(ei.value)


def test_load_from_directory():
    """按目录加载示例插件（模拟 TG_OPS_PATH）。"""
    d = os.path.join(REPO, "examples", "plugins")
    for fn in ("softplus_plugin.py", "gelu_exact_plugin.py"):
        ops_plugin.load_plugin_file(os.path.join(d, fn))
    names = ops_plugin.registered_names()
    assert "softplus" in names and "gelu_exact" in names

    # 数值对齐
    x_np = np.array([[-1.5, 0.0, 2.0]], np.float32)
    x = _t(x_np)
    y = x.gelu_exact()
    tx = torch.from_numpy(x_np.copy())
    assert np.allclose(y.to_numpy(), torch.nn.functional.gelu(tx).numpy(), atol=1e-6)


def test_list_ops_shape():
    ops_plugin.register("listed_op", lambda x: x, doc="给清单看的")
    rows = ops_plugin.list_ops()
    row = next(r for r in rows if r["name"] == "listed_op")
    assert row["kind"] == "functional"
    assert row["doc"].startswith("给清单看的")
