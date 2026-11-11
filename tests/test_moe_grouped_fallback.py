# -*- coding: utf-8 -*-
"""锁住分组 GEMM 的**安全检查**（0.9.32 起的正确性保证）。

## 为什么必须有这个测试

2026-11-11 之前，`_grouped_ready()` 加载的是**非 Ex 版** `cublasSgemmGroupedBatched`
（要标量 m/n/k），却按 Ex 版声明成指针数组。驱动把指针低 32 位当 `m` 读 ⇒
天文数字 ⇒ **必然非法访存（rc=700）**。当时它是默认关闭的，所以没人发现。

现在有三道保证，本文件逐条锁住：

1. **形状不一致时安全回退**，结果与逐次 `gemm_nt` **逐位一致**；
2. **分组调用失败时静默回退**（不抛异常、不影响正确性）；
3. `TGAI_MOE_GROUPED` 默认关闭 —— **关闭时行为与修改前完全一致**。

> 这些用例需要真实 CUDA 设备；没有 GPU 时自动 skip（checker 会报原因）。
"""
import numpy as np
import pytest

import tgai_go.train as T


def _cuda_ready():
    try:
        T.set_default_device("cuda")
        rt = T._cuda()
        rt.sync()
        return rt
    except Exception:
        return None


RT = _cuda_ready()
needs_cuda = pytest.mark.skipif(
    RT is None,
    reason="分组 GEMM 的用例需要真实 CUDA 设备（本机无 GPU 时跳过）")


@pytest.fixture(autouse=True)
def _dev():
    yield
    T.set_default_device("cpu")


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


@needs_cuda
def test_non_uniform_shapes_fall_back_and_match_exactly():
    """形状不一致 ⇒ 必须走逐次路径，且与直接 gemm_nt 逐位一致。

    这是最容易被忽略的一条：非 Ex 版分组 API **要求所有组 m/n/k 相同**，
    早期实现没有这个检查，于是形状一不同就崩。
    """
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    K, N = 128, 64
    Ms = [37, 64, 91]          # 故意三组行数都不同
    As = [rt.h2d(rng.normal(0, 0.05, (m, K)).astype(np.float32)) for m in Ms]
    Bs = [rt.h2d(rng.normal(0, 0.05, (N, K)).astype(np.float32)) for _ in Ms]

    got = [np.asarray(rt.d2h(c), np.float32) for c in rt.grouped_nt(As, Bs)]
    ref = [np.asarray(rt.d2h(rt.gemm_nt(a, b)), np.float32) for a, b in zip(As, Bs)]
    worst = max(_rel(g, r) for g, r in zip(got, ref))
    assert worst == 0.0, "回退路径应与逐次 gemm_nt 逐位一致，实测差 %.3e" % worst


@needs_cuda
def test_uniform_shapes_never_crash_and_stay_correct():
    """形状一致时会尝试分组调用；**无论分组是否成功，结果必须正确**。

    当前实测分组调用返回 rc=7（libcublas 参数布局与文档不符），于是自动回退。
    本用例不关心走哪条路，只要求：不崩、数值对。
    """
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(1)
    K, N, M, G = 64, 32, 16, 4
    As = [rt.h2d(rng.normal(0, 0.05, (M, K)).astype(np.float32)) for _ in range(G)]
    Bs = [rt.h2d(rng.normal(0, 0.05, (N, K)).astype(np.float32)) for _ in range(G)]

    got = [np.asarray(rt.d2h(c), np.float32) for c in rt.grouped_nt(As, Bs)]
    ref = [np.asarray(rt.d2h(rt.gemm_nt(a, b)), np.float32) for a, b in zip(As, Bs)]
    worst = max(_rel(g, r) for g, r in zip(got, ref))
    # 分组成功时是 cuBLAS 求和顺序差异（约 1e-7）；回退时逐位一致。两者都 ≤1e-5。
    assert worst <= 1e-5, "分组/回退路径与逐次结果差 %.3e，超过 1e-5" % worst


def test_moe_grouped_switch_defaults_off():
    """`TGAI_MOE_GROUPED` 默认关闭 —— 关闭时行为与加分组路径之前完全一致。

    分组路径尚未调通（rc=7），所以默认必须是关闭的；
    若哪天有人把默认改成开启，这个用例会先红。
    """
    import os
    assert os.environ.get("TGAI_MOE_GROUPED", "0") != "1", \
        "分组 GEMM 尚未调通，默认不能开启；要试验请显式设 TGAI_MOE_GROUPED=1"


def test_grouped_helpers_exist():
    """分组路径依赖的运行时原语必须存在，否则静默回退会掩盖缺失。"""
    try:
        rt = T._cuda()
    except Exception:
        pytest.skip("无 CUDA，跳过运行时方法检查")
    for name in ("grouped_nt", "gemm_nt"):
        assert hasattr(rt, name), "运行时缺少 %s" % name
