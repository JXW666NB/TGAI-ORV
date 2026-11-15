# -*- coding: utf-8 -*-
"""回归：惰性转置（`.T`）之后，所有算子都必须看到**转置后**的数据。

## 背景（2026-11-15 修的真实 BUG）

`_transpose_cuda` 为省一次拷贝不物化，只给结果打 `_lazy_T` 标记。
当时只有 `matmul`（走 `gemm_nt`）和 `to_numpy` 认这个标记，**其余算子直接读原始缓冲**，
于是 `x.T + 1`、`x.T.reshape(...)`、`x.T.sum(axis=1)`、甚至 `x.T.T` 全都静默算错。

修法：把 `_gpu` 做成属性，读缓冲前先物化（见 `tools/fix_lazy_transpose.py`）。
本文件锁死这些行为，防止退化。
"""
import numpy as np
import pytest

import tgai_go.train as T


def _has_cuda():
    try:
        T._cuda()
        return True
    except Exception:      # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="需要 CUDA（BUG 只在 CUDA 路径）")


def _pair():
    a = np.arange(6, dtype=np.float32).reshape(3, 2)
    T.set_default_device("cuda")
    x = T.from_numpy(a)
    return a, x, x.T


def test_to_numpy_is_transposed():
    a, _x, t = _pair()
    assert np.allclose(t.to_numpy(), a.T)


def test_reshape_after_transpose():
    a, _x, t = _pair()
    assert np.allclose(t.reshape(6).to_numpy(), a.T.reshape(6))


def test_contiguous_then_reshape():
    a, _x, t = _pair()
    assert np.allclose(t.contiguous().reshape(6).to_numpy(), a.T.reshape(6))


def test_elementwise_add_scalar():
    a, _x, t = _pair()
    assert np.allclose((t + 1.0).to_numpy(), a.T + 1.0)


def test_elementwise_mul_scalar():
    a, _x, t = _pair()
    assert np.allclose((t * 2.0).to_numpy(), a.T * 2.0)


def test_neg():
    a, _x, t = _pair()
    assert np.allclose((-t).to_numpy(), -a.T)


def test_exp():
    a, _x, t = _pair()
    assert np.allclose(t.exp().to_numpy(), np.exp(a.T), rtol=1e-6)


def test_sum_axis_keeps_transposed_shape():
    a, _x, t = _pair()
    got = t.sum(axis=1)
    assert tuple(got.shape) == (a.T.shape[0],)
    assert np.allclose(got.to_numpy(), a.T.sum(axis=1))


def test_double_transpose_is_identity():
    a, _x, t = _pair()
    assert np.allclose(t.T.to_numpy(), a)


def test_add_two_tensors():
    a, _x, t = _pair()
    y = T.from_numpy(np.ones((2, 3), np.float32))
    assert np.allclose((t + y).to_numpy(), a.T + np.ones((2, 3), np.float32))


def test_matmul_still_correct():
    a, _x, t = _pair()                    # t: [2, 3]
    y = T.from_numpy(np.ones((3, 4), np.float32))
    assert np.allclose(T.matmul(t, y).to_numpy(), a.T @ np.ones((3, 4), np.float32))


def test_grad_through_transpose():
    a = np.arange(6, dtype=np.float32).reshape(3, 2)
    T.set_default_device("cuda")
    x = T.from_numpy(a)
    x.requires_grad = True
    (x.T * x.T).sum().backward()
    assert np.allclose(x.grad.to_numpy(), 2 * a)


def test_matmul_with_transposed_rhs_does_not_copy():
    """`.T` 的省拷贝优化必须还在：`a @ b.T` 不应把 b.T 物化。"""
    T.set_default_device("cuda")
    rt = T._cuda()
    # 必须绕开"精确 GEMM"路径（M,N<=64 且 K<=16 走 numpy 对齐的朴素内核，那条本就要物化），
    # 否则测不到 gemm_nt 的省拷贝
    a = T.from_numpy(np.random.rand(129, 256).astype(np.float32))
    b = T.from_numpy(np.random.rand(97, 256).astype(np.float32))
    bt = b.T
    calls = {"n": 0}
    orig = rt.transpose2d

    def spy(*args, **kw):
        calls["n"] += 1
        return orig(*args, **kw)

    rt.transpose2d = spy
    try:
        out = T.matmul(a, bt)
        out.to_numpy()
    finally:
        rt.transpose2d = orig
    assert calls["n"] == 0, "matmul 不该为 .T 物化（会白拷一次），实际调用 %d 次" % calls["n"]
    assert np.allclose(out.to_numpy(), a.to_numpy() @ b.to_numpy().T)
