# -*- coding: utf-8 -*-
"""arch 降级链（2026-11-21）：设备比 NVRTC 新时用低一档 PTX 让驱动 JIT。

背景：RTX 5090 是 sm_120，而机器上装的是 CUDA 12.4 的 NVRTC —— 精确拱形被拒：
`nvrtc: error: invalid value for --gpu-architecture (-arch)`，ORV 直接起不来。
PTX 向前兼容，所以按代际降级即可。
"""
import pytest

from tgai_go.train.cuda import _arch_candidates


def test_sm_120_falls_back_through_known_generations():
    got = _arch_candidates("compute_120")
    assert got[0] == "compute_120"
    assert got == ["compute_120", "compute_100", "compute_90", "compute_86",
                   "compute_80", "compute_75", "compute_70"]


def test_old_arch_only_lists_older_ones():
    assert _arch_candidates("compute_86") == [
        "compute_86", "compute_80", "compute_75", "compute_70"]


def test_lowest_arch_has_no_fallback():
    assert _arch_candidates("compute_70") == ["compute_70"]


@pytest.mark.parametrize("bad", ["", "sm_90", "compute_", "compute_abc", None])
def test_garbage_returns_itself(bad):
    """输入不合法时不要构造出奇怪的候选，原样返回让上层报真实错误。"""
    assert _arch_candidates(bad) == [bad]


def test_no_duplicates():
    for a in ("compute_120", "compute_100", "compute_90", "compute_80"):
        got = _arch_candidates(a)
        assert len(got) == len(set(got))
        assert got == sorted(got, key=lambda s: -int(s.split("_")[1]))
