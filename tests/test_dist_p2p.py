# -*- coding: utf-8 -*-
"""P2P 全归约的**无卡**测试：只验 world=1 空操作、参数校验与错误信息。

真正的跨卡通信无法在 CI 上跑（需要两张同机 GPU），但下面这些不变量可以先钉住，
免得租卡时才发现是拼写/校验错误而不是性能问题。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

from tgai_go.train import dist_p2p as P2P  # noqa: E402


def test_world_one_is_noop(monkeypatch):
    monkeypatch.setenv("TGAI_DP_WORLD", "1")
    monkeypatch.setenv("TGAI_DP_P2P", "1")
    ar = P2P.P2PAllReduce()
    assert ar.enabled is False
    # 空操作不得抛异常，也不得碰 CUDA
    ar.attach(None)
    ar.all_reduce_grads(None)


def test_p2p_off_by_default_even_with_world_two(monkeypatch):
    monkeypatch.setenv("TGAI_DP_WORLD", "2")
    monkeypatch.delenv("TGAI_DP_P2P", raising=False)
    ar = P2P.P2PAllReduce()
    assert ar.enabled is False, "未显式开启 TGAI_DP_P2P 时必须保持关闭（走旧的主机路径）"


def test_bad_rank_message_says_range(monkeypatch):
    monkeypatch.setenv("TGAI_DP_WORLD", "2")
    monkeypatch.setenv("TGAI_DP_RANK", "3")
    with pytest.raises(P2P.P2PError) as ei:
        P2P.P2PAllReduce()
    msg = str(ei.value)
    assert "越界" in msg and "修法" in msg


def test_rendezvous_dir_is_created(monkeypatch):
    monkeypatch.setenv("TGAI_DP_JOB", "unit_test_job_xyz")
    d = P2P.rendezvous_dir()
    assert os.path.isdir(d)
    assert "tgai_dp_unit_test_job_xyz" in d


def test_p2p_requested_parsing(monkeypatch):
    for v, want in (("1", True), ("true", True), ("ON", True), ("0", False), ("", False)):
        monkeypatch.setenv("TGAI_DP_P2P", v)
        assert P2P.p2p_requested() is want
