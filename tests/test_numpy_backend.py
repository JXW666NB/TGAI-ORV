# -*- coding: utf-8 -*-
"""NumPy 参考后端的语义测试 —— 它是接口的「活文档」，必须自己是对的。

这些测试不需要 GPU，CI 上能跑；也用来证明「核心接口定义是自洽的」。
"""
import numpy as np
import pytest

from tgai_go.train import backend
from tgai_go.train.backends import numpy_ref


@pytest.fixture()
def rt():
    return numpy_ref.make()


def A(rt, a):
    """把 numpy 数组送进后端（显式传 rt，避免引用到全局的 fixture 函数名）。"""
    return rt.h2d(np.asarray(a, dtype=np.float32))


def N(buf):
    return np.asarray(buf.arr, dtype=np.float32)


def test_h2d_d2h_roundtrip(rt):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert np.array_equal(N(rt.h2d(x)), x)
    assert np.array_equal(rt.d2h(rt.clone(A(rt, x))), x)


def test_sgemm_and_nt_tn(rt):
    a = np.random.default_rng(0).normal(size=(4, 5)).astype(np.float32)
    b = np.random.default_rng(1).normal(size=(5, 3)).astype(np.float32)
    assert np.allclose(N(rt.sgemm(A(rt, a), A(rt, b))), a @ b, rtol=1e-6, atol=1e-6)
    # gemm_nt: A[M,K] @ B[N,K]^T
    assert np.allclose(N(rt.gemm_nt(A(rt, a), A(rt, b.T))), a @ b, rtol=1e-6, atol=1e-6)
    # gemm_tn: A[K,M]^T @ B[K,N]
    assert np.allclose(N(rt.gemm_tn(A(rt, a.T), A(rt, b))), a @ b, rtol=1e-6, atol=1e-6)


def test_matmul_shape_error_has_fix(rt):
    with pytest.raises(ValueError) as ei:
        rt.sgemm(A(rt, np.zeros((3, 4), np.float32)), A(rt, np.zeros((5, 2), np.float32)))
    msg = str(ei.value)
    assert "形状" in msg and "修法" in msg, msg


def test_softmax_matches_numpy_and_is_stable(rt):
    x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]], dtype=np.float32)
    got = N(rt.softmax_lastdim(A(rt, x)))
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    want = e / e.sum(axis=-1, keepdims=True)
    assert np.allclose(got, want, rtol=1e-6, atol=1e-7)
    assert np.isfinite(got).all()          # 大数值不溢出


def test_sum_all_and_read_scalar(rt):
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    s = rt.sum_all(A(rt, x))
    assert np.isclose(rt.read_scalar(s), 10.0)
    with pytest.raises(ValueError) as ei:
        rt.read_scalar(A(rt, x))
    assert "标量" in str(ei.value)


def test_sum_axis(rt):
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.allclose(N(rt.sum_axis(A(rt, x), 0)), x.sum(0))
    assert np.allclose(N(rt.sum_axis(A(rt, x), 1, keepdims=True)), x.sum(1, keepdims=True))


def test_reshape_permute_swap_broadcast(rt):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert N(rt.reshape(A(rt, x), (3, 2))).shape == (3, 2)
    assert np.array_equal(N(rt.permute(A(rt, x), (1, 0))), x.T)
    m = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(N(rt.swap_last2(A(rt, m))), m.T)
    assert np.array_equal(N(rt.transpose2d(A(rt, m))), m.T)
    b = N(rt.broadcast_to(A(rt, np.ones((1, 4), np.float32)), (3, 4)))
    assert b.shape == (3, 4) and np.allclose(b, 1.0)


def test_inplace_ops(rt):
    x = A(rt, np.ones((2, 2), np.float32))
    y = A(rt, np.full((2, 2), 2.0, np.float32))
    rt.scale_inplace(x, 3.0)
    assert np.allclose(N(x), 3.0)
    rt.add_(x, y)
    assert np.allclose(N(x), 5.0)
    rt.sub_scale_(x, y, 2.0)
    assert np.allclose(N(x), 1.0)


def test_unbroadcast(rt):
    g = A(rt, np.ones((3, 4), np.float32))
    assert np.allclose(N(rt.unbroadcast(g, (1, 4))), np.full((1, 4), 3.0))
    assert np.allclose(N(rt.unbroadcast(g, (1, 1))), np.full((1, 1), 12.0))


def test_div_by_zero_names_fix(rt):
    with pytest.raises(ZeroDivisionError) as ei:
        rt.div_scalar(A(rt, np.ones((2,), np.float32)), 0)
    assert "eps" in str(ei.value)


def test_backend_report_tiers():
    r = backend.report(numpy_ref.make())
    assert r["name"] == "numpy_ref"
    assert r["core"][1] == len(backend.core_methods())
    assert r["fallback"][1] == len(backend.fallback_methods())
    assert r["core"][0] >= 30, r["core"]
    # 模板不追求训得起来，但必须能被人看出「还缺什么」
    assert r["missing_core"], "参考后端应当仍有缺口，如实报告"
