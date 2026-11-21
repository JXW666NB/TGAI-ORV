# -*- coding: utf-8 -*-
"""重叠归约的**硬自检**：没真归约就当场炸，不许静默少做工作。

2026-11-21 实测（2×4090, `--dp 2`, mode=host, 13 桶）：
`[dp-ov] comm=0.00ms exposed=0.03ms pending_zeroed=True` 看起来完全健康，
但普通 `all_reduce_grads` 一次都没被调用、两 rank 的 gnorm 相差 1e-2
⇒ 梯度根本没被归约，数据并行退化成「没开 DP」，而 loss 照样下降。
这种 bug 只能靠逐位对拍或这类硬自检发现。
"""
import pytest

from tgai_go.train import v3_pretrain as V


def test_zero_comm_with_multiple_worlds_raises():
    """world=2 且 comm≈0 ⇒ 必炸（这就是实测到的静默不归约）。"""
    with pytest.raises(RuntimeError) as e:
        V._check_overlap_stats({"comm_ms": 0.0, "pending_zeroed": True}, 2)
    msg = str(e.value)
    assert "没有真的发生" in msg
    assert "TG_DP_OVERLAP=0" in msg          # 必须给出可执行的退路


def test_tiny_comm_also_raises():
    """0.5ms 以内一律当没搬东西（3 GiB 归约不可能这么快）。"""
    with pytest.raises(RuntimeError):
        V._check_overlap_stats({"comm_ms": 0.4}, 2)


def test_real_comm_passes():
    V._check_overlap_stats({"comm_ms": 143.2}, 2)     # 不抛


def test_single_world_is_exempt():
    """world=1 没有归约，comm=0 是正常的。"""
    V._check_overlap_stats({"comm_ms": 0.0}, 1)
    V._check_overlap_stats(None, 2)                   # 会话没生效（None）也不炸


def test_bad_values_are_treated_as_zero():
    with pytest.raises(RuntimeError):
        V._check_overlap_stats({"comm_ms": None}, 2)
    with pytest.raises(RuntimeError):
        V._check_overlap_stats({}, 2)
