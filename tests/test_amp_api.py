# -*- coding: utf-8 -*-
"""T.amp() 可编程 AMP 的契约（CPU 可跑部分）。

判据：
1. `amp_state()` 在没有 CUDA 时不能炸（要给出 on=False）；
2. `amp()` 的入参校验（dtype 只支持 fp16）；
3. `GradScaler` 的算术：scale_loss / unscale_ / 非有限值跳过 step / 增长；
4. PyT 迁移代码里 `scaler.step(opt); scaler.update()` 的调用形状要能照抄。
"""
import numpy as np
import pytest

import tgai_go.train as T


@pytest.fixture(autouse=True)
def _force_cpu_default_device():
    """别的用例可能把默认设备留在 cuda；这里强制回 cpu 并复原，避免相互污染。"""
    try:
        T.set_default_device("cpu")
    except Exception:
        pass
    yield
    try:
        T.set_default_device("cpu")
    except Exception:
        pass


def test_amp_state_does_not_crash_without_cuda():
    st = T.amp_state()
    assert isinstance(st, dict) and "on" in st


def test_amp_accepts_bf16_and_rejects_garbage():
    """bf16 现在是合法 dtype（V100 上运行期会回退 fp32 并警告，不是报错）。"""
    assert T.amp(dtype="bf16").dtype == "bf16"
    with pytest.raises(NotImplementedError):
        T.amp(dtype="int4")


def test_amp_context_is_a_context_manager():
    a = T.amp()
    assert hasattr(a, "__enter__") and hasattr(a, "__exit__")


# ---------------- GradScaler（纯 numpy 可测） ----------------
class _P(object):
    def __init__(self, g):
        self.grad = np.asarray(g, np.float32)
        self.requires_grad = True


class _Opt(object):
    def __init__(self, params, lr=0.1):
        self.params = params
        self.lr = lr
        self.done = 0

    def step(self):
        self.done += 1
        for p in self.params:
            p.grad = None

    def zero_grad(self):
        for p in self.params:
            p.grad = None


def test_scaler_scale_and_unscale_roundtrip():
    """存的是**已缩放**的梯度（真实流程就是这样），unscale 后应回到原值。"""
    s = T.GradScaler(init_scale=1024.0)
    p = _P(np.array([2.0 * 1024.0, -4.0 * 1024.0]))
    scaled = s.scale_loss(T.tensor(np.array(2.0, np.float32)))
    assert abs(float(scaled.to_numpy() if hasattr(scaled, "to_numpy") else scaled) - 2048.0) < 1e-3
    bad = s.unscale_([p])
    assert bad is False
    assert np.allclose(np.asarray(p.grad, np.float32), [2.0, -4.0], rtol=1e-6), p.grad


def test_scaler_skips_step_on_nonfinite():
    s = T.GradScaler(init_scale=8.0, backoff_factor=0.5)
    p = _P(np.array([np.inf, 1.0]))
    opt = _Opt([p])
    ok = s.step(opt)
    assert ok is False
    assert opt.done == 0, "有非有限梯度时不能调用 opt.step()"
    assert s.scale == 4.0, "跳过后 scale 应减半"
    assert s.skipped == 1


def test_scaler_grows_after_interval():
    s = T.GradScaler(init_scale=1.0, growth_factor=2.0, growth_interval=2)
    for _ in range(2):
        p = _P(np.array([1.0]))
        opt = _Opt([p])
        assert s.step(opt) is True
    assert s.scale == 2.0
    assert s.steps == 2


def test_scaler_pytorch_style_calls_work():
    """PyT 迁移代码的调用形状：scale_loss → backward → step(opt) → update()。"""
    s = T.GradScaler()
    p = _P(np.array([0.5]))
    opt = _Opt([p])
    loss = s.scale_loss(T.tensor(np.array(1.0, np.float32)))
    assert loss is not None
    assert s.step(opt) is True
    assert s.update() > 0
