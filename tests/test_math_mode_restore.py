# -*- coding: utf-8 -*-
"""`ns_math_mode_end` 必须复位到"引擎当前认为的模式"，不能硬编码 PEDANTIC。

背景（2026-11-21）：`--amp tf32` 的实现复用了 Muon 的 `ns_math_mode_begin/end`。
若复位写死 PEDANTIC，则 `--amp tf32` + `TG_MUON_NS_GEMM=tf32` + `--opt muon` 的组合下，
NS 的一次 begin/end 会把全局 TF32 静默关掉，而 `_tf32["on"]` 仍是 True ⇒ 之后所有 GEMM
退回 PEDANTIC，且没有任何报错。
"""
import types

import pytest

from tgai_go.train.cuda import (CUBLAS_PEDANTIC_MATH, CUBLAS_TF32_TENSOR_OP_MATH,
                                CudaRuntime)


class _FakeCublas:
    def __init__(self):
        self.modes = []

    def cublasSetMathMode(self, handle, mode):
        self.modes.append(mode)
        return 0


def _fake_rt(tf32_on):
    rt = object.__new__(CudaRuntime)
    rt.cublas = _FakeCublas()
    rt.cu = object()                 # 非 None 即视为"上下文已就绪"
    rt.handle = 1
    rt._chk_blas = lambda rc, what: None
    rt._activate = lambda: None          # begin() 会先激活上下文，假对象要有这个方法
    rt._tf32 = {"on": tf32_on, "ok": True}
    return rt


def test_restore_goes_back_to_tf32_when_amp_tf32_is_on():
    rt = _fake_rt(True)
    rt.ns_math_mode_end()
    assert rt.cublas.modes == [CUBLAS_TF32_TENSOR_OP_MATH]


def test_restore_goes_back_to_pedantic_when_amp_tf32_is_off():
    rt = _fake_rt(False)
    rt.ns_math_mode_end()
    assert rt.cublas.modes == [CUBLAS_PEDANTIC_MATH]


def test_restore_defaults_to_pedantic_when_state_is_missing():
    """没经过 tf32_enable 的运行时（例如没有 _tf32 属性）不能因此崩或停在 TF32。"""
    rt = _fake_rt(False)
    del rt._tf32
    rt.ns_math_mode_end()
    assert rt.cublas.modes == [CUBLAS_PEDANTIC_MATH]


def test_begin_then_end_keeps_tf32_when_enabled():
    """完整轮回：AMP tf32 开着时，begin→end 出来仍然是 TF32（不会把全局关掉）。"""
    rt = _fake_rt(True)
    assert rt.ns_math_mode_begin() is True
    rt.ns_math_mode_end()
    assert rt.cublas.modes == [CUBLAS_TF32_TENSOR_OP_MATH,
                              CUBLAS_TF32_TENSOR_OP_MATH]
