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


# ---------------------------------------------------------------------------
# 编译器选择（2026-11-20：CI windows 腿连续红的两个原因）
# ---------------------------------------------------------------------------

def test_tgai_cc_pointing_at_zig_is_detected_as_zig(monkeypatch, tmp_path):
    """`TGAI_CC=<...>/zig.exe` 必须被识别成 zig。

    原来只认 tcc，其它一律当 "unix"，于是调用 `zig.exe -shared ...`
    （缺了必需的 `cc` 子命令）而失败 —— 而 zig 正是文档推荐的零配置路径。
    """
    zig = tmp_path / "zig.exe"
    zig.write_bytes(b"")
    monkeypatch.setenv("TGAI_CC", str(zig))
    cc, kind = ops_cpu.find_compiler()
    assert kind == "zig", "TGAI_CC 指向 zig 时应返回 kind='zig'，实际 %r" % (kind,)
    assert cc == str(zig)


def test_windows_clang_is_replaced_by_zig(monkeypatch):
    """Windows 上 clang→MSVC 目标**不导出符号**，产物 CTYPES 找不到插件函数。

    这是 CI windows 腿第二个红的根因（第一个是 -fPIC）。只要有 zig 可用，
    就应该改用它；Linux 上不受影响（默认全导出）。
    """
    if sys.platform != "win32":
        pytest.skip("只在 Windows 上有意义")
    monkeypatch.delenv("TGAI_CC", raising=False)
    monkeypatch.setattr(ops_cpu.shutil, "which",
                        lambda n: r"C:\fake\LLVM\bin\clang.EXE" if n == "clang" else None)
    monkeypatch.setattr(ops_cpu, "_find_zig", lambda: r"C:\fake\ziglang\zig.exe")
    cc, kind = ops_cpu.find_compiler()
    assert kind == "zig", "Windows 上应避开 clang，实际 kind=%r cc=%r" % (kind, cc)
    assert cc.lower().endswith("zig.exe")


def test_explicit_tgai_cc_wins_over_zig_fallback(monkeypatch, tmp_path):
    """用户显式指定 TGAI_CC 时**始终尊重**，不要被 zig 回退抢走。"""
    tcc = tmp_path / "tcc.exe"
    tcc.write_bytes(b"")
    monkeypatch.setenv("TGAI_CC", str(tcc))
    cc, kind = ops_cpu.find_compiler()
    assert cc == str(tcc)
    assert kind == "tcc"