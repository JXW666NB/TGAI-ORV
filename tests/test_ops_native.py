# -*- coding: utf-8 -*-
"""多后端原生算子注册：自动降级链（纯 CPU 可测的部分）。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

from tgai_go.train import ops_native  # noqa: E402


def test_python_fallback_is_always_available():
    ops_native.register("_t_softplus", {"python": lambda x: (x.exp() + 1.0).log()})
    st = ops_native.status()["_t_softplus"]
    assert st["chosen"] == "python"
    assert "python" in ops_native.supported()


def test_missing_backend_raises_with_fix_hint():
    with pytest.raises(ValueError) as ei:
        ops_native.register("_t_only_cuda", {"cuda": "nope.cu"})
    assert "修法" in str(ei.value)


def test_empty_impls_rejected():
    with pytest.raises(ValueError):
        ops_native.register("_t_empty", {})


def test_non_callable_non_cuda_impl_rejected():
    # 本机没有 cuda 时，若实现给的是字符串（源码），不能当 Python 函数用
    if "cuda" in ops_native.supported():
        pytest.skip("本机有 CUDA，这条不适用")
    with pytest.raises(ValueError) as ei:
        ops_native.register("_t_str", {"hip": "x.hip"})
    assert "修法" in str(ei.value)
