# -*- coding: utf-8 -*-
"""int8 分块 AdamW 的一致性实测（CPU，进 CI）。

**实测结论（2026-11-08）**：无误差反馈的 int8 动量不可用；int8+**fp16 残差**的误差反馈才达标。
| 配置 | B/参数 | 参数偏差 | 余弦 | loss 偏差 |
| 无 EF | 2.13 | 5.26 | 0.327 | 6.6e-2 |
| EF-int8 残差 | 4.25 | 12.5 | 0.135 | 2.8e-1 |
| EF-fp16 残差 | 6.13 | 3.4e-5 | 1.0 | 1.2e-7 |
所以 CI 只把 **EF-fp16** 当作"必须达标"，其余作为反例（防止有人删掉残差还宣称达标）。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train.int8_adamw import Fp32AdamW, Int8AdamW  # noqa: E402


def _problem(seed=0, n=4096):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 1.0, n).astype(np.float32),
            rng.normal(0, 1.0, n).astype(np.float32))


def _run(opt, w0, target, steps=100):
    w = w0.copy()
    losses = []
    for _ in range(steps):
        g = (w - target).astype(np.float32)
        losses.append(float(0.5 * np.sum((w - target) ** 2)))
        opt.step({"w": w}, {"w": g})
    losses.append(float(0.5 * np.sum((w - target) ** 2)))
    return w, np.asarray(losses)


def _cmp(opt, w0, target):
    a, la = _run(Fp32AdamW(), w0, target)
    b, lb = _run(opt, w0, target)
    rel = np.abs(b - a).max() / max(1e-9, np.abs(a).max())
    cos = float(np.dot(b - w0, a - w0) /
                max(1e-12, np.linalg.norm(b - w0) * np.linalg.norm(a - w0)))
    dev = float((np.abs(lb - la) / np.maximum(np.abs(la), 1e-9))[::10].max())
    return rel, cos, dev


def test_ef_fp16_meets_criteria():
    """EF-fp16 残差：参数偏差 ≤1e-3、余弦 ≥0.999、loss 偏差 ≤1%。"""
    w0, tgt = _problem()
    rel, cos, dev = _cmp(Int8AdamW(resid_dtype="float16"), w0, tgt)
    print("\nEF-fp16: 偏差=%.2e 余弦=%.6f loss=%.2e" % (rel, cos, dev))
    assert rel <= 1e-3 and cos >= 0.999 and dev <= 1e-2


def test_without_ef_is_rejected():
    """没有误差反馈就必须**不达标** —— 这条用来防止"偷偷删残差还宣称等效"。"""
    w0, tgt = _problem()
    rel, cos, _ = _cmp(Int8AdamW(), w0, tgt)
    print("\n无EF: 偏差=%.2e 余弦=%.6f（本应很差）" % (rel, cos))
    assert cos < 0.999, "无误差反馈竟然达标了？先确认测试是否失真"


def test_int8_residual_is_rejected():
    """int8 残差省内存但精度崩（实测余弦 0.13~0.98）⇒ 不许用它冒充达标。"""
    w0, tgt = _problem()
    _, cos, _ = _cmp(Int8AdamW(resid_dtype="int8"), w0, tgt)
    print("\nEF-int8: 余弦=%.6f（本应 <0.999）" % cos)
    assert cos < 0.999


def test_bytes_per_param():
    assert abs(Int8AdamW().bytes_per_param() - 2.125) < 1e-3
    assert abs(Int8AdamW(resid_dtype="float16").bytes_per_param() - 6.125) < 1e-3


def test_no_nan_1000_steps():
    rng = np.random.default_rng(2)
    w0 = rng.normal(0, 1, 1024).astype(np.float32)
    tgt = rng.normal(0, 1, 1024).astype(np.float32)
    w, losses = _run(Int8AdamW(resid_dtype="float16", lr=1e-2), w0, tgt, steps=1000)
    assert np.isfinite(w).all() and losses[-1] < losses[0]
def test_fp16_moments_meets_criteria():
    """**推荐方案**：m/v 存 fp16（4 B/参数）—— 2.97B 只占 11.9 GB，且精度达标。"""
    from tgai_go.train.int8_adamw import Fp16AdamW
    w0, tgt = _problem()
    rel, cos, dev = _cmp(Fp16AdamW(), w0, tgt)
    print("\nfp16 m/v: 偏差=%.2e 余弦=%.6f loss=%.2e" % (rel, cos, dev))
    assert rel <= 1e-3 and cos >= 0.999 and dev <= 1e-2
    assert abs(Fp16AdamW().bytes_per_param() - 4.0) < 1e-9


def test_fp16_moments_no_nan_1000():
    from tgai_go.train.int8_adamw import Fp16AdamW
    rng = np.random.default_rng(3)
    w0 = rng.normal(0, 1, 1024).astype(np.float32)
    tgt = rng.normal(0, 1, 1024).astype(np.float32)
    w, losses = _run(Fp16AdamW(lr=1e-2), w0, tgt, steps=1000)
    assert np.isfinite(w).all() and losses[-1] < losses[0]