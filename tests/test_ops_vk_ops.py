# -*- coding: utf-8 -*-
"""Vulkan 后端测试。

**端到端部分走子进程**：AMD 驱动在同一进程里反复建/销毁 device 会导致进程崩溃
（实测 pytest 直接段错误），所以把真实 dispatch 放进子进程，崩溃只会让这一条失败，
不会拖垮整个测试套件。
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import ops_vk  # noqa: E402
from tgai_go.train import ops_vk_ops as V  # noqa: E402

CHILD = r'''
import sys
sys.path.insert(0, {repo!r})
import numpy as np
import tgai_go.train as T
from tgai_go.train import ops_vk_ops as V
FWD = {fwd!r}
BWD = {bwd!r}
V.register_elementwise("vk_relu_child", FWD, BWD)
x = np.random.default_rng(0).normal(size=(512,)).astype(np.float32)
t = T.tensor(x.copy(), requires_grad=True)
y = getattr(T.functional, "vk_relu_child")(t)
fw = float(np.abs(np.asarray(y.to_numpy()) - np.maximum(x, 0)).max())
g = np.random.default_rng(1).normal(size=(512,)).astype(np.float32)
(y * T.tensor(g)).sum().backward()
ref = np.where(x > 0, g, 0.0).astype(np.float32)
bw = float(np.abs(np.asarray(t.grad) - ref).max())
print("FWD", fw)
print("BWD", bw)
'''

FWD = """
#version 450
layout(local_size_x = 64) in;
layout(binding = 0) buffer A { float a[]; };
layout(binding = 1) buffer O { float o[]; };
void main() { uint i = gl_GlobalInvocationID.x; if (i < o.length()) o[i] = a[i] > 0.0 ? a[i] : 0.0; }
"""
BWD = """
#version 450
layout(local_size_x = 64) in;
layout(binding = 0) buffer G { float g[]; };
layout(binding = 1) buffer A { float a[]; };
layout(binding = 2) buffer D { float d[]; };
void main() { uint i = gl_GlobalInvocationID.x; if (i < d.length()) d[i] = a[i] > 0.0 ? g[i] : 0.0; }
"""


def test_bind_point_constant_is_compute():
    """曾经写成 4（非法 bind point）：内核与描述符都绑不上，写被静默丢弃、输出全 0。"""
    from tgai_go.train import ops_vk_compute as M
    assert M.BIND_COMPUTE == 1


def test_spv_of_rejects_garbage():
    with pytest.raises(FileNotFoundError):
        V.spv_of("/nonexistent/x.spv")


def test_glsl_without_compiler_has_clear_message():
    if V._glslang() is not None or os.environ.get("TGAI_GLSLANG"):
        pytest.skip("本机有 SPIR-V 编译器，这条不适用")
    with pytest.raises(RuntimeError) as ei:
        V.compile_glsl(FWD)
    assert "SPIR-V" in str(ei.value) and "修法" in str(ei.value)


def _vk_ready():
    """可用性检查也放子进程：父进程一旦加载 vulkan-1.dll，退出时可能段错误。"""
    code = ("import sys; sys.path.insert(0, %r);"
            "from tgai_go.train import ops_vk;"
            "print('OK' if (ops_vk.available() and ops_vk.devices()) else 'NO')"
            % os.path.join(REPO, "python"))
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                       capture_output=True, text=True,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return r.returncode == 0 and r.stdout.strip().endswith("OK")


def test_vulkan_relu_end_to_end_in_subprocess():
    if not _vk_ready():
        pytest.skip("本机没有可用的 Vulkan 设备")
    if V._glslang() is None and not os.environ.get("TGAI_GLSLANG"):
        pytest.skip("没有 SPIR-V 编译器（设 TGAI_GLSLANG）")
    code = CHILD.format(repo=os.path.join(REPO, "python"), fwd=FWD, bwd=BWD)
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                       capture_output=True, text=True,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert r.returncode == 0, "子进程失败（可能是驱动崩溃）：\n%s\n%s" % (r.stdout[-800:],
                                                                 r.stderr[-1500:])
    vals = {}
    for line in r.stdout.splitlines():
        if line.startswith("FWD "):
            vals["fwd"] = float(line.split()[1])
        elif line.startswith("BWD "):
            vals["bwd"] = float(line.split()[1])
    assert vals.get("fwd") == 0.0, "前向不是逐位精确：%r" % (vals,)
    assert vals.get("bwd") == 0.0, "反向不是逐位精确：%r" % (vals,)
