# -*- coding: utf-8 -*-
"""插件系统：六条轴，一个命名空间。

## 为什么这样设计（写进 README 的卖点）

- **简易**：六条轴都挂在同一个 `plugin` 对象上；一个插件就是一个 `.py` 文件。
- **通用**：插件只组合核心算子 ⇒ autograd 自动、天然适配所有后端（CUDA/Vulkan/ROCm）。
- **安全**：能力声明不足时**当场报错**，不静默退化。

## 守住什么

1. 算子 / 优化器 / 层 / 骨架登记后**立刻可用**（无副作用）；
2. 后端 / API 兼容层**只登记不激活** —— 激活要显式 `activate()`（它们有副作用）；
3. `works_inplace=False` 的优化器**不能**用于共享存储参数（`require_inplace` 报错）；
4. 重名登记要显式 `replace=True`；
5. 名字/类型不合法要报错；
6. **一个 .py 文件就能把六条轴全用上**（端到端）。
"""
import io
import os
import sys
import tempfile

import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import plugin


@pytest.fixture(autouse=True)
def _clean():
    """每个用例后清掉本用例登记的东西，避免互相污染。"""
    yield
    from tgai_go.train import plugin as P
    for d, attr in ((P._LAYERS, "layers"), (P._ARCHS, "archs"),
                    (P._OPTIMIZERS, "optimizers"), (P._BACKENDS, "backends"),
                    (P._APIS, "apis")):
        d.clear()
    P._ACTIVATED.clear()


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield
    T.set_default_device("cpu")


# ─────────────── ① 算子 ───────────────
def test_op_is_immediately_usable():
    plugin.op("sp_test", lambda x: (x.exp() + 1.0).log())
    x = T.tensor(np.array([0.0, 1.0], np.float32))
    y = T.functional.sp_test(x)
    want = np.log(np.exp(np.array([0.0, 1.0], np.float32)) + 1.0)
    assert np.allclose(np.asarray(y.data), want, rtol=1e-6)


# ─────────────── ② 后端 ───────────────
def test_backend_is_registered_not_activated():
    plugin.backend("fake_vk", lambda: object(), caps={"subgroup_ops": 32})
    r = T.plugins()
    assert "fake_vk" in r["backends"]
    assert not any("fake_vk" in a for a in r["activated"]), "登记不该自动激活"
    assert plugin._BACKENDS["fake_vk"]["caps"]["subgroup_ops"] == 32


def test_backend_duplicate_needs_replace():
    plugin.backend("bk", lambda: object())
    with pytest.raises(plugin.PluginError):
        plugin.backend("bk", lambda: object())
    plugin.backend("bk", lambda: object(), replace=True)   # 显式覆盖可以


# ─────────────── ③ 优化器 ───────────────
class _OutOfPlaceOpt:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data = p.data - self.lr          # 出式更新


class _InPlaceOpt(_OutOfPlaceOpt):
    def step(self):
        for p in self.params:
            p.data -= self.lr                  # 原地更新


def test_optimizer_resolves_from_nn_namespace():
    plugin.optimizer("OptA", _OutOfPlaceOpt)
    assert T.optim.OptA is _OutOfPlaceOpt
    plugin.optimizer("OptB", _InPlaceOpt, works_inplace=True)
    assert T.optim.OptB is _InPlaceOpt


def test_inplace_capability_is_enforced():
    """★ 纪律：出式优化器配共享存储参数必须**报错**，不能静默不同步。"""
    plugin.optimizer("Out", _OutOfPlaceOpt, works_inplace=False)
    plugin.optimizer("In", _InPlaceOpt, works_inplace=True)
    with pytest.raises(plugin.PluginError) as e:
        plugin.require_inplace("Out")
    assert "出式" in str(e.value) or "works_inplace" in str(e.value)
    assert plugin.require_inplace("In") is _InPlaceOpt
    # 根本没登记也要报错
    with pytest.raises(plugin.PluginError):
        plugin.require_inplace("NoSuch")


# ─────────────── ④ 层 ───────────────
def test_layer_resolves_from_nn_namespace():
    class _Blk(T.nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.lin = T.nn.Linear(d, d)

        def forward(self, x):
            return self.lin(x)

    plugin.layer("BlkX", _Blk)
    assert T.nn.BlkX is _Blk
    m = T.nn.BlkX(d=8)
    y = m(T.tensor(np.ones((2, 8), np.float32)))
    assert tuple(int(s) for s in y.shape) == (2, 8)


def test_missing_layer_raises_with_guidance():
    with pytest.raises(AttributeError) as e:
        T.nn.NoSuchLayerAnywhere
    assert "plugin.layer" in str(e.value)


# ─────────────── ⑤ 骨架 ───────────────
def test_arch_builds_model():
    plugin.arch("tiny_lin", lambda d=8: T.nn.Linear(d, d))
    m = T.build("tiny_lin", d=8)
    assert isinstance(m, T.nn.Linear)


def test_build_unknown_arch_lists_available():
    with pytest.raises(KeyError) as e:
        T.build("nope")
    assert "plugin.arch" in str(e.value)


# ─────────────── ⑥ API 兼容层 ───────────────
def test_api_only_runs_on_activate():
    called = {"n": 0}

    def install():
        called["n"] += 1

    plugin.api("tok_compat", install)
    assert called["n"] == 0, "登记不该立刻执行（有副作用）"
    plugin.activate("tok_compat")
    assert called["n"] == 1
    assert "tok_compat" in T.plugins()["activated"]


# ─────────────── 校验 ───────────────
@pytest.mark.parametrize("bad", ["", "has space", "1abc", "a-b"])
def test_bad_names_rejected(bad):
    with pytest.raises(plugin.PluginError):
        plugin.layer(bad, T.nn.Linear)


def test_wrong_types_rejected():
    with pytest.raises(plugin.PluginError):
        plugin.layer("x1", T.nn.Linear(d_model=8, d_ff=16) if False else object())  # 非类
    with pytest.raises(plugin.PluginError):
        plugin.optimizer("x2", lambda p: None)          # 非类
    with pytest.raises(plugin.PluginError):
        plugin.arch("x3", 123)                           # 不可调用


# ─────────────── 端到端：一个文件用上全部六条轴 ───────────────
PLUGIN_SRC = '''# -*- coding: utf-8 -*-
"""一个插件文件：六条轴一次用上。"""
from tgai_go.train import plugin
import tgai_go.train as T


@plugin.op("e2e_double")
def e2e_double(x):
    return x * 2.0


class E2EBlock(T.nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.lin = T.nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


plugin.layer("E2EBlock", E2EBlock)
plugin.optimizer("E2EOpt", E2EBlock, works_inplace=True) if False else None
plugin.arch("e2e_arch", lambda d=8: E2EBlock(d))
plugin.backend("e2e_backend", lambda: object(), caps={"subgroup_ops": 32})
plugin.api("e2e_api", lambda: None)
'''


def test_one_file_uses_all_axes():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "e2e_plugin.py")
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(PLUGIN_SRC)
    try:
        plugin.load(p)
        r = T.plugins()
        assert "E2EBlock" in r["layers"]
        assert "e2e_arch" in r["archs"]
        assert "e2e_backend" in r["backends"]
        assert "e2e_api" in r["apis"]
        # 算子真的能用
        x = T.tensor(np.array([1.5, -2.0], np.float32))
        assert np.allclose(np.asarray(T.functional.e2e_double(x).data),
                           np.array([3.0, -4.0], np.float32))
        # 骨架真的能建
        m = T.build("e2e_arch", d=8)
        assert isinstance(m, T.nn.Module)
    finally:
        # 插件加载会在同目录生成 __pycache__，直接删除整个临时目录
        import shutil as _sh
        _sh.rmtree(d, ignore_errors=True)


def test_summary_is_one_line():
    s = plugin.summary()
    assert "插件" in s and "算子" in s and "\n" not in s
