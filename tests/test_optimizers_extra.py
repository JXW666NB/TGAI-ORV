# -*- coding: utf-8 -*-
"""LAMB 优化器：与独立参考实现逐步一致 + trust ratio 不变量 + CUDA 明确拒绝。

参照物是**测试里另写一遍**的公式实现（不复用被测代码），这样公式理解错了能被抓到。
另一个强判据是 trust ratio 的定义本身：
    一步更新量的范数  ||Δp|| ≈ lr · ||p||   （wd=0 时）
这条不依赖任何实现细节，写错了必然违反。
"""
import numpy as np
import pytest

import tgai_go.train as T


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def _ref_lamb_step(data, m, v, t, g, lr, b1, b2, eps, wd):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    u = (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps) + wd * data
    pn, un = np.linalg.norm(data), np.linalg.norm(u)
    trust = pn / un if (pn > 0 and un > 0) else 1.0
    return data - lr * trust * u, m, v


def test_lamb_matches_reference_step_by_step():
    lr, b1, b2, eps, wd = 0.05, 0.9, 0.999, 1e-6, 0.01
    p = T.nn.Parameter(np.array([1.0, -2.0, 0.5, 4.0], np.float32))
    opt = T.optim.LAMB([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)

    data = np.array([1.0, -2.0, 0.5, 4.0])
    m, v = np.zeros(4), np.zeros(4)
    rng = np.random.default_rng(0)
    for t in range(1, 6):
        g = rng.normal(0, 0.3, 4)
        p.grad = g.astype(np.float32)
        opt.step()
        data, m, v = _ref_lamb_step(data, m, v, t, g, lr, b1, b2, eps, wd)
        got = np.asarray(p.to_numpy(), np.float64)
        assert np.allclose(got, data, rtol=1e-5, atol=1e-6), "第 %d 步不一致" % t


def test_trust_ratio_makes_update_norm_equal_lr_times_param_norm():
    """trust ratio 的定义：||Δp|| == lr * ||p||（wd=0、且范数都非零）。"""
    lr = 0.1
    p = T.nn.Parameter(np.array([3.0, 4.0], np.float32))     # ||p|| = 5
    opt = T.optim.LAMB([p], lr=lr, weight_decay=0.0)
    before = np.asarray(p.to_numpy(), np.float64)
    p.grad = np.array([0.7, -0.2], np.float32)
    opt.step()
    after = np.asarray(p.to_numpy(), np.float64)
    assert np.isclose(np.linalg.norm(after - before), lr * np.linalg.norm(before), rtol=1e-5)


def test_weight_decay_shrinks_further():
    p1 = T.nn.Parameter(np.array([3.0, 4.0], np.float32))
    p2 = T.nn.Parameter(np.array([3.0, 4.0], np.float32))
    g = np.array([0.7, -0.2], np.float32)
    o1 = T.optim.LAMB([p1], lr=0.1, weight_decay=0.0)
    o2 = T.optim.LAMB([p2], lr=0.1, weight_decay=0.5)
    p1.grad = g.copy()
    p2.grad = g.copy()
    o1.step()
    o2.step()
    n1 = np.linalg.norm(np.asarray(p1.to_numpy(), np.float64))
    n2 = np.linalg.norm(np.asarray(p2.to_numpy(), np.float64))
    assert n2 < n1, "weight_decay 应该让参数缩得更多（%s vs %s）" % (n2, n1)


def test_lamb_refuses_cuda_params_loudly():
    p = T.nn.Parameter(np.array([1.0, 2.0], np.float32))
    p.device = "cuda"                      # 只伪装属性，不碰真设备
    with pytest.raises(NotImplementedError) as ei:
        T.optim.LAMB([p], lr=0.1)
    msg = str(ei.value)
    assert "LAMB" in msg and "修法" in msg and "AdamW" in msg


def test_lamb_is_registered_on_optim_namespace():
    assert hasattr(T.optim, "LAMB")
    assert T.optim.LAMB is T.LAMB if hasattr(T, "LAMB") else True
