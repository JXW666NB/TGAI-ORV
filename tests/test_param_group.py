# -*- coding: utf-8 -*-
"""共享存储参数（`T.ParameterGroup`）。

## 为什么需要

MoE 的专家权重若连续存成 `[G, N, K]`，一次 `SgemmStridedBatched` 就能算完全部专家；
否则每次都要先把 G 份权重复制成连续缓冲 —— 实测打包开销比省下的还多
（docs/MOE_MODERN_ARCH.md 3.1）。共享存储参数是"权重连续存"的数据模型基础。

## 契约（不遵守会静默出错，所以必须测住）

1. 视图真的与底层共享内存（写底层 ⇒ 视图立刻可见）；
2. `parameters()` **只返回底层那一块**，绝不返回视图；
3. 只有 `works_inplace=True` 的优化器能用于本组（出式更新会断开视图）；
4. 优化器若拿到了视图，必须报错。

## 平台

CPU 可用；**CUDA 本版不支持**（`GpuBuf` 无 offset 字段），文档与代码都写明，不静默出错。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import plugin
from tgai_go.train.param_group import SharedStorageError


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield
    T.set_default_device("cpu")


@pytest.fixture(autouse=True)
def _clean_plugins():
    yield
    from tgai_go.train import plugin as P
    P._OPTIMIZERS.clear()


class _InPlace:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)

    def step(self):
        for p in self.params:
            p.data -= 0.1          # 原地


class _OutOfPlace:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)

    def step(self):
        for p in self.params:
            p.data = p.data - 0.1  # 出式：会断开视图


def test_views_share_memory():
    g = T.ParameterGroup([(4, 3), (4, 3)])
    assert g.assert_shared() is True
    base = np.asarray(g.packed.data)
    base.ravel()[0] = 42.0
    assert float(np.asarray(g.views[0].data).ravel()[0]) == 42.0, "写底层后视图应立刻可见"
    base.ravel()[12] = -7.0
    assert float(np.asarray(g.views[1].data).ravel()[0]) == -7.0, "第二个视图也应共享"


def test_layout_is_tightly_packed():
    g = T.ParameterGroup([(2, 2), (3,)])
    assert g.packed.numel() == 7
    assert [tuple(int(x) for x in v.shape) for v in g.views] == [(2, 2), (3,)]


def test_parameters_excludes_views():
    """★ 契约：优化器只能拿到底层那一块，视图绝不能出现在 parameters() 里。"""
    g = T.ParameterGroup([(2, 3), (2, 3)])
    ps = g.parameters()
    assert len(ps) == 1, "应只返回底层一块，实际 %d 个" % len(ps)
    assert ps[0] is g.packed
    view_ids = {id(v) for v in g.views}
    assert not (view_ids & {id(p) for p in ps}), "视图不该出现在 parameters() 里"


def test_inplace_capability_is_required():
    """★ 出式优化器必须被拒绝 —— 否则权重与视图会不同步且极难查。"""
    plugin.optimizer("_Inp", _InPlace, works_inplace=True)
    plugin.optimizer("_Out", _OutOfPlace, works_inplace=False)
    g = T.ParameterGroup([(2, 2), (2, 2)])
    assert g.optimizer_class("_Inp") is _InPlace
    with pytest.raises(plugin.PluginError) as e:
        g.optimizer_class("_Out")
    assert "works_inplace" in str(e.value) or "出式" in str(e.value)


def test_optimizer_holding_views_is_rejected():
    g = T.ParameterGroup([(2, 2), (2, 2)])
    bad = _InPlace(g.views)                 # 错：喂了视图
    with pytest.raises(SharedStorageError) as e:
        g.assert_optimizer_ok(bad)
    assert "视图" in str(e.value)
    good = _InPlace(g.parameters())         # 对：喂底层
    assert g.assert_optimizer_ok(good) is True


def test_out_of_place_step_actually_breaks_and_is_caught():
    """实证"出式更新会断开视图"—— 这正是必须拒绝它的原因。"""
    g = T.ParameterGroup([(2, 2)])
    g.packed.data = g.packed.data - 0.1      # 出式赋值（模拟错误优化器）
    # numpy 的 `arr = arr - 0.1` 会换成新数组；视图仍指向旧缓冲
    still_shared = np.shares_memory(np.asarray(g.packed.data),
                                    np.asarray(g.views[0].data))
    assert not still_shared, "出式赋值应断开共享，本用例用来证明风险真实存在"


def test_rejects_bad_input():
    with pytest.raises(SharedStorageError):
        T.ParameterGroup([])
    with pytest.raises(SharedStorageError):
        T.ParameterGroup([(2, 2)], dtype=np.float16)


def test_repr_is_informative():
    g = T.ParameterGroup([(2, 2), (2, 2)])
    s = repr(g)
    assert "2 个视图" in s and "8 元素" in s
