# -*- coding: utf-8 -*-
"""CPU 后端 C 插件：编译 → ctypes → autograd（纯 CPU）。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

from tgai_go.train import ops_cpu  # noqa: E402

SRC = r"""
void r_r_fwd(const float* a, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] > 0.f ? a[i] : 0.f;
}
void r_r_bwd(const float* g, const float* a, float* da, int n) {
    for (int i = 0; i < n; ++i) da[i] = a[i] > 0.f ? g[i] : 0.f;
}
"""


def _has_cc():
    cc, _ = ops_cpu.find_compiler()
    return cc is not None


def test_find_compiler_reports_something_or_none():
    cc, kind = ops_cpu.find_compiler()
    assert (cc is None) or isinstance(cc, str)


def test_missing_symbol_message():
    if not _has_cc():
        pytest.skip("本机没有 C 编译器")
    h = ops_cpu.load_source(SRC, name="cpu_sym_test")
    with pytest.raises(RuntimeError) as ei:
        ops_cpu._sym(h, "not_a_symbol_xyz")
    assert "修法" in str(ei.value)


def test_wrong_c_source_reports_compiler_log():
    if not _has_cc():
        pytest.skip("本机没有 C 编译器")
    with pytest.raises(RuntimeError) as ei:
        ops_cpu.load_source("void f( { broken", name="cpu_bad_test")
    assert "编译" in str(ei.value) and "修法" in str(ei.value)


def test_cpu_plugin_fwd_bwd_is_correct():
    if not _has_cc():
        pytest.skip("本机没有 C 编译器")
    import tgai_go.train as T
    h = ops_cpu.load_source(SRC, name="cpu_relu_test")
    ops_cpu.register_elementwise("cpu_relu_test", h, "r_r_fwd", "r_r_bwd")
    x = np.random.default_rng(0).normal(size=(16,)).astype(np.float32)
    t = T.tensor(x.copy(), requires_grad=True)
    y = getattr(T.functional, "cpu_relu_test")(t)
    assert np.allclose(np.asarray(y.to_numpy()), np.maximum(x, 0))
    (y * T.tensor(np.ones(16, np.float32))).sum().backward()
    assert np.allclose(np.asarray(t.grad), (x > 0).astype(np.float32))
