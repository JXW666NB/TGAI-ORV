# -*- coding: utf-8 -*-
"""C 端插件优先：原生库、C ABI 包装、JSON 声明式清单。

## 设计优先级（用户明确要求）

**插件优先 C 端，其次 Python。** 三条通道，按"原生程度"排列：

| 通道 | 形态 | 热路径 |
| --- | --- | --- |
| `plugin.load_manifest(json)` | **纯声明式**：用户编译原生库 + 写 JSON，**零 Python** | 原生 |
| `plugin.op_c(lib, symbol)` | 预编译库 + C ABI，Python 只做一次绑定 | 原生 |
| `plugin.op_native(source, entry)` | CUDA C 源码，**ORV 用 NVRTC 现场编译** | 原生 |
| `plugin.op(fn)` | 组合已有算子 | Python 胶水 |

`.py` 从来不是热路径 —— 它只在加载时执行一次。这条要在文档里讲清，避免误解。

## 守住什么

1. 缺库 / 缺符号 / 坏 JSON / 坏参数声明，都要报错且给出「修法」；
2. `args` 声明（`ptr:N` / `f32:N` / `int:N`）正确映射；
3. 原生库**只加载一次**（同一路径复用句柄）；
4. 清单里的 `library` 相对清单文件目录解析；
5. 安全声明存在（加载原生库 = 执行任意机器码）。
"""
import ctypes
import ctypes.util as _ctypes_util
import io
import json
import os
import shutil
import sys
import tempfile

import pytest

import tgai_go.train as T
from tgai_go.train import plugin, plugin_manifest


@pytest.fixture(autouse=True)
def _clean():
    yield
    plugin._NATIVE_OPS.clear()
    plugin._LIBS.clear()
    plugin._LIBS_META.clear()


def _system_lib():
    """拿一个确定存在的系统 C 库做 ABI 链路验证（不依赖本机编译器）。"""
    cand = _ctypes_util.find_library("msvcrt") or _ctypes_util.find_library("c")
    if not cand:
        pytest.skip("本机找不到可用的系统 C 库")
    return ctypes.CDLL(cand)


# ─────────────── op_c：C ABI 包装 ───────────────
def test_op_c_wraps_exported_symbol():
    lib = _system_lib()
    got = {}

    def args(v):
        got["v"] = int(v)
        return [int(v)]

    f = plugin.op_c("my_abs", lib, "abs", args=args, restype=ctypes.c_int)
    f(-7)
    assert got["v"] == -7, "参数映射应被调用"
    assert "my_abs" in plugin.native_info()["registered"]


def test_op_c_missing_symbol_explains_fix():
    lib = _system_lib()
    with pytest.raises(plugin.PluginError) as e:
        plugin.op_c("bad", lib, "definitely_not_a_symbol_xyz")
    msg = str(e.value)
    assert "extern" in msg or "nm -D" in msg, "错误信息要说明怎么排查导出名，实际：%s" % msg


def test_load_lib_missing_file_explains_fix():
    with pytest.raises(plugin.PluginError) as e:
        plugin.load_lib("no_such_lib_xyz.so")
    assert "修法" in str(e.value)


def test_load_lib_is_cached():
    path = _ctypes_util.find_library("msvcrt") or _ctypes_util.find_library("c")
    if not path:
        pytest.skip("本机找不到系统 C 库")
    a = plugin.load_lib(path)
    b = plugin.load_lib(path)
    assert a is b, "同一路径应复用句柄，不重复加载"


# ─────────────── 参数声明解析 ───────────────
def test_parse_args_spec_maps_types():
    assert plugin_manifest.parse_args_spec(["f32:0", "int:1"], [2.5, 9]) == [2.5, 9]
    assert plugin_manifest.parse_args_spec(["raw:0"], ["x"]) == ["x"]
    assert plugin_manifest.parse_args_spec([], [1, 2]) == []


def test_parse_args_spec_rejects_bad_input():
    with pytest.raises(ValueError):
        plugin_manifest.parse_args_spec(["bogus:0"], [1])
    with pytest.raises(ValueError):
        plugin_manifest.parse_args_spec(["ptr:5"], [1])      # 越界
    with pytest.raises(ValueError):
        plugin_manifest.parse_args_spec(["nocolon"], [1])
    with pytest.raises(ValueError):
        plugin_manifest.parse_args_spec(["f32:x"], [1])


def test_parse_args_spec_ptr_requires_device_buffer():
    with pytest.raises(ValueError) as e:
        plugin_manifest.parse_args_spec(["ptr:0"], [object()])
    assert "CUDA" in str(e.value)


# ─────────────── JSON 清单 ───────────────
def test_manifest_missing_file_and_bad_json():
    with pytest.raises(plugin.PluginError):
        plugin.load_manifest("no_such_manifest.json")

    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "bad.json")
        with io.open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write("{not json")
        with pytest.raises(plugin.PluginError) as e:
            plugin.load_manifest(p)
        assert "JSON" in str(e.value)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_manifest_requires_core_fields():
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "m.json")
        with io.open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"name": "x"}, f)
        with pytest.raises(plugin.PluginError) as e:
            plugin.load_manifest(p)
        assert "api_version" in str(e.value) or "library" in str(e.value)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_manifest_binds_ops_end_to_end():
    """端到端：JSON + 原生库 → 算子可调用（用系统 C 库的 abs 做真实绑定）。"""
    libname = _ctypes_util.find_library("msvcrt") or _ctypes_util.find_library("c")
    if not libname:
        pytest.skip("本机找不到系统 C 库")
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "myops.json")
        with io.open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump({
                "name": "myops",
                "version": "0.1.0",
                "api_version": "1.0",
                "library": os.path.abspath(libname),
                "ops": [{"name": "abs_via_json", "symbol": "abs",
                         "args": ["int:0"], "restype": "int"}],
            }, f)
        info = plugin.load_manifest(p)
        assert info["name"] == "myops"
        assert "abs_via_json" in plugin.native_info()["registered"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ─────────────── op_native：NVRTC 现场编译 ───────────────
def test_op_native_validates_source():
    with pytest.raises(plugin.PluginError) as e:
        plugin.op_native("x", "", "k")
    assert "source" in str(e.value) or "源码" in str(e.value)
    with pytest.raises(plugin.PluginError):
        plugin.op_native("x", "extern \"C\" __global__ void k(){}", "")


def test_op_native_without_cuda_errors_clearly():
    """无 GPU 环境下要明确报错（不是静默返回错误结果）。"""
    f = plugin.op_native(
        "k_no_gpu", 'extern "C" __global__ void k_no_gpu(float* y){}', "k_no_gpu",
        grid=(1, 1, 1), block=(1, 1, 1), args=lambda y: [0])
    T.set_default_device("cpu")
    try:
        with pytest.raises(plugin.PluginError) as e:
            f(T.tensor([1.0]))
        assert "CUDA" in str(e.value)
    finally:
        T.set_default_device("cpu")


def test_native_info_reports_channels():
    info = plugin.native_info()
    assert set(info) >= {"registered", "compiled", "libraries"}


def test_docs_states_c_first_and_security():
    """文档必须写明「C 端优先」与「加载原生库等于执行任意机器码」。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, "docs", "PLUGIN_NATIVE.md")
    assert os.path.isfile(p), "必须有 docs/PLUGIN_NATIVE.md"
    s = io.open(p, encoding="utf-8").read()
    assert "C 端" in s or "原生" in s
    assert "任意机器码" in s, "必须写明加载原生库的安全风险"
