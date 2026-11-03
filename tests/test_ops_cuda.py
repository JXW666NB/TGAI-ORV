# -*- coding: utf-8 -*-
"""CUDA 插件：用户写 .cu 由引擎编译、或加载已编译 cubin。

CPU 可跑的部分（错误信息）在这里钉住；真实内核在 GPU 上跑（TG_RUN_CUDA_TESTS=1）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

from tgai_go.train import ops_cuda  # noqa: E402

GELU_SRC = r"""
extern "C" __global__ void gelu_fwd(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    float t = 0.7978845608f * (x + 0.044715f * x * x * x);
    out[i] = 0.5f * x * (1.0f + tanhf(t));
}
extern "C" __global__ void gelu_bwd(const float* g, const float* a, float* da, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    float t = 0.7978845608f * (x + 0.044715f * x * x * x);
    float th = tanhf(t);
    float dt = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
    da[i] = g[i] * (0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th) * dt);
}
"""


def test_load_cubin_missing_file_says_how_to_fix():
    with pytest.raises(FileNotFoundError) as ei:
        ops_cuda.load_cubin("no_such_plugin.cubin")
    assert "cubin" in str(ei.value) and "修法" in str(ei.value)


@pytest.mark.skipif(not os.environ.get("TG_RUN_CUDA_TESTS"),
                    reason="需要 GPU：设 TG_RUN_CUDA_TESTS=1")
def test_cuda_source_plugin_gelu_matches_reference():
    np = pytest.importorskip("numpy")
    import tgai_go.train as T

    T.set_default_device("cuda")
    rt = T._cuda()
    mod = ops_cuda.load_source(GELU_SRC, cu_name="gelu_plugin_test.cu")
    ops_cuda.register_elementwise("gelu_plugin_test", mod, "gelu_fwd", "gelu_bwd")

    x = np.random.default_rng(0).normal(size=(256,)).astype(np.float32)
    t = T.tensor(x.copy(), requires_grad=True)
    y = getattr(T.functional, "gelu_plugin_test")(t)
    rt.sync()
    got = np.asarray(y.to_numpy(), np.float64)
    ref = 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-6

    g = np.random.default_rng(1).normal(size=(256,)).astype(np.float32)
    (y * T.tensor(g)).sum().backward()
    rt.sync()
    gx = np.asarray(t.grad.to_numpy(), np.float64)
    eps = 1e-3
    f = lambda v: 0.5 * v * (1 + np.tanh(0.7978845608 * (v + 0.044715 * v ** 3)))
    num = np.array([(f(np.float64(v + eps)) - f(np.float64(v - eps))) / (2 * eps) for v in x])
    denom = np.abs(g * num).max() + 1e-12
    assert np.abs(gx - g * num).max() / denom < 1e-3
