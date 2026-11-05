# -*- coding: utf-8 -*-
"""GradScaler 的**逻辑**测试（纯 CPU，不需要 GPU）。

为什么单独测：AMP 在 0.5B 上先是「梯度下溢学不动」、再是「缩放太大溢出尖峰」，
最后靠这个 scaler 才跑通 1200 步。它的增长/回退/跳步语义一旦写错，
训练会静默变差（不报错、只是学不动），所以必须有用例锁住。

这里把设备相关部分（有限性检查、原地除缩放）换成桩，只验状态机。
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import amp_mlp  # noqa: E402


class _FakeRT:
    """假运行时：只记录 scale_inplace 的调用，不碰 CUDA。"""

    def __init__(self):
        self.unscaled = []

    def scale_inplace(self, buf, s):
        self.unscaled.append(float(s))


def _scaler(**kw):
    s = amp_mlp.GradScaler(**kw)
    # 桩：不碰 CUDA。有限性检查与原地除缩放都换成假实现，只验状态机。
    s._finite_flag = True
    s._grads_finite = lambda params: s._finite_flag
    s._rt_for_test = _FakeRT()
    amp_mlp._rt = lambda: s._rt_for_test
    return s


def test_begin_publishes_scale():
    s = _scaler(init_scale=1024.0)
    s.begin()
    assert amp_mlp.AMP_GRAD_SCALE == 1024.0


def test_growth_after_interval():
    s = _scaler(init_scale=8.0, growth=2.0, growth_interval=3, max_scale=1e9)
    # 有限 → 前 3 次只在计数，第三次后翻倍
    for _ in range(3):
        assert s.end([]) is True
    assert s.scale == 16.0
    assert s.skipped == 0


def test_backoff_on_nonfinite():
    s = _scaler(init_scale=1024.0, backoff=0.5)
    s._finite_flag = False
    assert s.end([]) is False          # 不可更新
    assert s.scale == 512.0
    assert s.skipped == 1
    assert s.good == 0                  # 计数清零


def test_backoff_floor_is_one():
    s = _scaler(init_scale=2.0, backoff=0.5)
    s._finite_flag = False
    s.end([])
    assert s.scale == 1.0
    s.end([])                           # 再降也不能低于 1
    assert s.scale == 1.0


def test_growth_respects_max_scale():
    s = _scaler(init_scale=1024.0, growth=2.0, growth_interval=1, max_scale=1024.0)
    for _ in range(5):
        s.end([])
    assert s.scale == 1024.0            # 到顶不再涨


def test_amp_default_off():
    """AMP 必须默认关闭：用户不显式打开就不该悄悄改数值。"""
    assert amp_mlp.enabled() is False