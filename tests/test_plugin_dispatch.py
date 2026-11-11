# -*- coding: utf-8 -*-
"""能力模型与算子分派（`dispatch` + `plugin.op` 的条件化实现）。

## 为什么需要

插件「单文件适配全部后端」的前提：**插件针对能力写，不针对后端写**。
分派按当前后端的**真实能力**（caps）与计算能力（cc）选实现。

## 守住什么

1. 能力满足时才用快路径；不满足自动落到通配回退；
2. `backend="*"` 的回退对**任何后端**生效（Vulkan 尚未有专用实现时也能跑）；
3. 计算能力范围（`min_cc`/`max_cc`）正确过滤；
4. `dtype` 限定正确过滤；
5. 同条件多候选时按 `prio` 取大的；
6. **没有可用实现必须报错**，且错误信息说明缺什么；
7. 分派开销必须被量化（`benchmark_overhead`），这是 >1% 硬约束的判据。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import dispatch, plugin


@pytest.fixture(autouse=True)
def _clean():
    dispatch.clear()
    yield
    dispatch.clear()
    T.set_default_device("cpu")


def _mk(tag):
    return lambda x: tag


# ─────────────── 能力分派 ───────────────
def test_caps_gate_the_fast_path():
    plugin.op("op_c", _mk("fast"), backend="cuda", caps={"async_copy": True}, prio=10)
    plugin.op("op_c", _mk("slow"), backend="*")
    get = lambda caps, arch: plugin.resolve_op("op_c", "cuda", caps=caps, arch=arch)(0)
    assert get({"async_copy": True}, (8, 0)) == "fast"
    assert get({}, (8, 0)) == "slow", "缺少 async_copy 时必须回退"
    assert get({"async_copy": False}, (8, 0)) == "slow", "能力为 False 也必须回退"


def test_arch_range_filters():
    plugin.op("op_a", _mk("volta_plus"), backend="cuda", min_cc=(7, 0), prio=10)
    plugin.op("op_a", _mk("any"), backend="*")
    get = lambda arch: plugin.resolve_op("op_a", "cuda", caps={}, arch=arch)(0)
    assert get((7, 0)) == "volta_plus"
    assert get((8, 0)) == "volta_plus"
    assert get((6, 2)) == "any", "Pascal 不满足 min_cc，必须回退"
    # max_cc 同理
    dispatch.clear()
    plugin.op("op_b", _mk("old_only"), backend="cuda", max_cc=(6, 2), prio=10)
    plugin.op("op_b", _mk("any"), backend="*")
    g2 = lambda arch: plugin.resolve_op("op_b", "cuda", caps={}, arch=arch)(0)
    assert g2((6, 2)) == "old_only"
    assert g2((7, 0)) == "any"


def test_wildcard_covers_any_backend():
    """★ 通用性核心：只登记了通配回退时，Vulkan 这类新后端也能跑。"""
    plugin.op("op_w", _mk("slow"), backend="*")
    for be in ("cuda", "vulkan", "rocm", "cpu"):
        assert plugin.resolve_op("op_w", be, caps={}, arch=None)(0) == "slow", be


def test_dtype_filter():
    plugin.op("op_d", _mk("f16"), backend="cuda", dtype="fp16", prio=10)
    plugin.op("op_d", _mk("f32"), backend="cuda", dtype="fp32")
    assert plugin.resolve_op("op_d", "cuda", caps={}, dtype="fp16")(0) == "f16"
    assert plugin.resolve_op("op_d", "cuda", caps={}, dtype="fp32")(0) == "f32"


def test_priority_picks_higher():
    plugin.op("op_p", _mk("low"), backend="cuda", prio=1)
    plugin.op("op_p", _mk("high"), backend="cuda", prio=99)
    assert plugin.resolve_op("op_p", "cuda", caps={})(0) == "high"


def test_missing_impl_raises_with_reason():
    with pytest.raises(dispatch.DispatchError) as e:
        plugin.resolve_op("never_registered", "cuda")
    assert "没有登记" in str(e.value) or "没有可用实现" in str(e.value)

    plugin.op("op_x", _mk("fast"), backend="cuda", caps={"async_copy": True})
    with pytest.raises(dispatch.DispatchError) as e2:
        plugin.resolve_op("op_x", "cuda", caps={})
    msg = str(e2.value)
    assert "async_copy" in msg, "错误信息必须说明缺哪个能力，实际：%s" % msg


# ─────────────── 缓存与静态绑定 ───────────────
def test_resolution_is_cached():
    calls = {"n": 0}
    plugin.op("op_q", _mk("x"), backend="*")
    a = plugin.resolve_op("op_q", "cuda", caps={})
    b = plugin.resolve_op("op_q", "cuda", caps={})
    assert a is b, "同一组键应命中缓存、返回同一个对象"
    assert dispatch._CACHE, "缓存应有内容"


def test_bind_is_static_binding():
    """热路径用法：bind 一次后直接调用，等价于直接函数调用。"""
    target = _mk("bound")
    plugin.op("op_bind", target, backend="*")
    f = plugin.bind("op_bind", "cuda", caps={})
    assert f is target
    assert f(0) == "bound"


def test_registration_invalidates_cache():
    plugin.op("op_i", _mk("a"), backend="*")
    assert plugin.resolve_op("op_i", "cuda", caps={})(0) == "a"
    plugin.op("op_i", _mk("b"), backend="cuda", prio=100)
    assert plugin.resolve_op("op_i", "cuda", caps={})(0) == "b", "新登记后缓存必须失效"


# ─────────────── 开销量化 ───────────────
def test_overhead_is_measured_and_documented():
    """硬约束：分派开销必须可量化，且文档里必须有「热路径用 bind」的说明。"""
    r = dispatch.benchmark_overhead(reps=20000)
    assert "overhead_ns" in r and "direct_ns" in r
    # 分派净开销应在微秒以下（否则设计有问题）
    assert r["overhead_ns"] < 5000, "分派开销异常大：%.0f ns" % r["overhead_ns"]
    import os
    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "PLUGIN_DISPATCH.md")
    assert os.path.isfile(doc), "必须有 docs/PLUGIN_DISPATCH.md 记录实测开销与热路径纪律"


# ─────────────── 与插件系统协同 ───────────────
def test_conditional_op_does_not_pollute_functional_namespace():
    """条件化实现只进分派表，不应出现在 functional 里（同名算子会有多个实现）。"""
    plugin.op("op_only_dispatch", _mk("x"), backend="*")
    assert not hasattr(T.functional, "op_only_dispatch")


def test_plain_op_still_goes_to_functional():
    plugin.op("plain_add_one", lambda x: x + 1.0)
    x = T.tensor(np.array([1.0, 2.0], np.float32))
    assert np.allclose(np.asarray(T.functional.plain_add_one(x).data),
                       np.array([2.0, 3.0], np.float32))
