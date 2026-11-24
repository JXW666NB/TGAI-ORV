# -*- coding: utf-8 -*-
"""audio-ops —— 用插件系统给 ORV 补上音频项目缺的算子（实测样例）。

## 背景（这不是杜撰的需求）

拿一个真实的 PyTorch 音频项目（DDSP 歌声合成，yuaz-sgr）做盘点：
它用到 87 个 torch API，ORV 同名同位的 37 个，**49 个没有**。
其中相当一部分是逐元素数学与损失函数 —— 本插件补的就是这一批。

## 两种补法，正好对应插件系统的两条通道

| 通道 | 用在什么上 | 例子 |
| --- | --- | --- |
| `plugin.op_native`（CUDA C 源码，NVRTC 现场编译）| 逐元素数学 | `sin / cos / sigmoid / softplus / log1p / log10 / leaky_relu / isfinite` |
| 纯 ORV 算子**组合** | 损失、形状、窗口 | `l1_loss / mse_loss / bce_loss / zeros_like / ones_like / pad_last / hann_window / magnitude` |

前者的热路径是**原生内核**，不是 Python 逐元素；后者只是搭积木，也全在设备上。

## 诚实边界（必须写明）

* 本插件只实现**前向**。要参与训练，每个算子还要登记 VJP —— 那是下一步。
* `stft / istft`（完整复数）、`linalg.pinv`、`unique`、`searchsorted` **不在本插件里**：
  它们需要真正的 FFT / 排序 / 线性代数内核，不是逐元素能解决的。逐元素内核解决不了它们。

## 用法

```python
import tgai_go.train as T
T.plugin.load("examples/plugins/audio_ops.py")
T.plugin.activate("audio-ops")
y = T.audio.sin(x)
w = T.audio.hann_window(1024)
loss = T.audio.l1_loss(a, b)
```
"""
import ctypes

import numpy as np

#: mode → 函数名（与内核里的分支一一对应）
MODES = {"sin": 0, "cos": 1, "sigmoid": 2, "softplus": 3,
         "log1p": 4, "log10": 5, "leaky_relu": 6, "isfinite": 7}

UNARY_SRC = r"""
extern "C" __global__ void unary_math(float* __restrict__ y,
                                      const float* __restrict__ x,
                                      int n, int mode, float slope) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float r = v;
    if (mode == 0) r = sinf(v);
    else if (mode == 1) r = cosf(v);
    else if (mode == 2) r = 1.0f / (1.0f + __expf(-v));
    else if (mode == 3) r = (v > 20.0f) ? v : log1pf(__expf(v));
    else if (mode == 4) r = log1pf(v);
    else if (mode == 5) r = log10f(v);
    else if (mode == 6) r = (v > 0.0f) ? v : slope * v;
    else if (mode == 7) r = (isfinite(v) ? 1.0f : 0.0f);
    y[i] = r;
}
"""


def _make_unary(plugin, T, name, mode, slope=0.01):
    """一个逐元素算子 = 一份 op_native 启动器 + 一层薄壳（负责分配输出）。"""
    launch = plugin.op_native(
        "audio_%s" % name, UNARY_SRC, "unary_math",
        grid=lambda x, y, n: ((int(n) + 255) // 256, 1, 1),
        block=(256, 1, 1),
        args=lambda x, y, n: [
            (int(y._gpu.ptr), ctypes.c_uint64),
            (int(x._gpu.ptr), ctypes.c_uint64),
            (int(n), ctypes.c_int),
            (int(mode), ctypes.c_int),
            (float(slope), ctypes.c_float)],
        replace=True,
        doc="audio-ops: %s（逐元素）" % name)

    def op(x):
        rt = T._cuda()
        y = rt.empty(tuple(x.shape), dtype=np.float32)
        launch(x, y, int(x.numel()))
        return T.Tensor._from_gpu(y, requires_grad=False)

    op.__name__ = name
    return op


def _composed(T):
    """只用 ORV 已有算子组合出来的部分。"""
    def zeros_like(x):
        return T.zeros(tuple(x.shape))

    def ones_like(x):
        return T.full(tuple(x.shape), 1.0)

    def pad_last(x, left, right, value=0.0):
        shape = list(x.shape)
        parts = []
        if left:
            parts.append(T.full(tuple(shape[:-1] + [int(left)]), float(value)))
        parts.append(x)
        if right:
            parts.append(T.full(tuple(shape[:-1] + [int(right)]), float(value)))
        return parts[0] if len(parts) == 1 else T.cat(parts, dim=-1)

    def l1_loss(a, b):
        d = T.abs(a - b)
        return d.sum() / float(d.numel())

    def mse_loss(a, b):
        d = a - b
        return (d * d).sum() / float(d.numel())

    def bce_loss(p, t, eps=1e-7):
        q = T.clamp(p, eps, 1.0 - eps)
        one = T.full(tuple(q.shape), 1.0)
        term = t * T.log(q) + (one - t) * T.log(one - q)
        return -(term.sum() / float(term.numel()))

    def hann_window(n):
        k = np.arange(int(n), dtype=np.float32)
        return T.tensor((0.5 - 0.5 * np.cos(2.0 * np.pi * k / float(n))).astype(np.float32))

    def magnitude(re, im):
        return T.sqrt(re * re + im * im)

    return {"zeros_like": zeros_like, "ones_like": ones_like, "pad_last": pad_last,
            "l1_loss": l1_loss, "mse_loss": mse_loss, "bce_loss": bce_loss,
            "hann_window": hann_window, "magnitude": magnitude}


def install():
    """装出 `T.audio` 命名空间并返回它。"""
    import types

    import tgai_go.train as T
    from tgai_go.train import plugin

    ops = {}
    for name, mode in MODES.items():
        ops[name] = _make_unary(plugin, T, name, mode)
    ops.update(_composed(T))

    ns = types.SimpleNamespace(**ops)
    T.audio = ns
    return ns


def _register():
    from tgai_go.train import plugin
    plugin.api("audio-ops", install, replace=True)


_register()
