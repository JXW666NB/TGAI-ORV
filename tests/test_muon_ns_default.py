# -*- coding: utf-8 -*-
"""Muon 的 Newton-Schulz 路径选择（2026-11-21：默认从 parity 慢路径改成 CUDA 快路径）。"""
import os

import numpy as np
import pytest

import tgai_go.train as T


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("TG_MUON_NS_FAST", "TG_MUON_NS_PARITY", "TG_MUON_NS_GEMM"):
        monkeypatch.delenv(k, raising=False)


def test_ns_gemm_defaults_to_fp32():
    """NS 的 GEMM 默认走 PEDANTIC fp32（逐位稳定）；TF32 必须显式开启。

    NS 是近似算法（5 步、eps=1e-7），官方 Muon 实现甚至用 bf16 跑它 ——
    但"默认不变"是这里的硬约束：精度相关的东西不能悄悄改。
    """
    assert T._muon_ns_gemm_mode() == "fp32"


@pytest.mark.parametrize("val", ["tf32", "1", "true", "YES", "on"])
def test_ns_gemm_tf32_opt_in(monkeypatch, val):
    monkeypatch.setenv("TG_MUON_NS_GEMM", val)
    assert T._muon_ns_gemm_mode() == "tf32"


@pytest.mark.parametrize("val", ["", "fp32", "乱写", "off"])
def test_ns_gemm_falls_back_to_fp32(monkeypatch, val):
    monkeypatch.setenv("TG_MUON_NS_GEMM", val)
    assert T._muon_ns_gemm_mode() == "fp32"


def test_default_is_fast():
    """没有环境变量时默认走快路径（cuBLAS + 并行归约）。

    这是 2026-11-21 修的那个「Muon 21.7 s/步」的主因：原来默认 False ⇒ 每个矩阵一次
    单线程 `sumsq_sequential` + `read_scalar` 同步，实测 [1024,4096] 单矩阵 101 ms。
    """
    assert T._muon_default_ns_fast() is True


@pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
def test_parity_env_forces_slow(monkeypatch, val):
    monkeypatch.setenv("TG_MUON_NS_PARITY", val)
    assert T._muon_default_ns_fast() is False


@pytest.mark.parametrize("val,expect", [
    ("1", True), ("true", True), ("on", True),
    ("0", False), ("false", False), ("off", False),
])
def test_explicit_fast_env_wins(monkeypatch, val, expect):
    """`TG_MUON_NS_FAST` 优先级最高，压过 parity。"""
    monkeypatch.setenv("TG_MUON_NS_FAST", val)
    monkeypatch.setenv("TG_MUON_NS_PARITY", "1")
    assert T._muon_default_ns_fast() is expect


def test_muon_accepts_none_and_bool():
    """`ns_fast=None` 必须被解析成布尔；显式 True/False 必须原样保留。"""
    p = np.zeros((4, 3), np.float32)
    for given in (None, True, False):
        opt = T.optim.Muon([p], lr=0.02, ns_fast=given)
        assert isinstance(opt.ns_fast, bool)
        if given is None:
            assert opt.ns_fast is T._muon_default_ns_fast()
        else:
            assert opt.ns_fast is given


def test_muon_parity_env_reaches_optimizer(monkeypatch):
    monkeypatch.setenv("TG_MUON_NS_PARITY", "1")
    p = np.zeros((4, 3), np.float32)
    assert T.optim.Muon([p], lr=0.02).ns_fast is False
