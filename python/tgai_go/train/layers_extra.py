# -*- coding: utf-8 -*-
"""用户自定义模型常用的补充层：Dropout、MaxPool/AvgPool（1d/2d）、BatchNorm（1d/2d）。

## 实现方式与已知边界（如实说明）

* **Dropout**：主机生成 mask（`>= p` 的伯努利）再与输入相乘，训练时按 `1/(1-p)` 缩放（inverted
  dropout，与 torch 一致）。CUDA 上 mask 通过 `rt.h2d` + `_from_gpu` 落到同设备，不额外占常驻显存。
  代价：每次前向一次 mask 的主机生成与拷贝（与张量同大小），大张量时比 cuDNN 的原生 dropout 慢。
* **池化**：用 `gather_flat`（阶段 B 已验证前/反向与 torch 逐位一致，反向是 scatter-add）实现，
  **CPU 与 CUDA 同一条路径**。窗口索引在主机侧按形状构造（Python 循环，大形状有主机开销）。
  - `AvgPool`：等价 `count_include_pad=True`（torch 默认）；`count_include_pad=False` 未实现。
  - `MaxPool`：需要窗口内的 argmax，CUDA 上要先 `to_numpy()`（**一次主机往返**）；并列最大值取
    “第一个”（与 torch 的 CPU 实现一致，CUDA 的 cuDNN 在并列时不保证）。
* **BatchNorm**：两遍法（先均值、再中心化后算方差），只用已有 `mean/mul/add`，CPU/CUDA 通用；
  训练时更新 running_mean/running_var，推理时用它们。`track_running_stats=False` 未实现。
"""
import numpy as np

from . import Module, Parameter, Tensor, _as_tensor, _cuda
from .indexing import gather_flat

__all__ = ["Dropout", "MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d",
           "BatchNorm1d", "BatchNorm2d"]


def _pair(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _like(np_arr, ref):
    """把 host 数组变成与 ref 同设备的张量（CUDA 走 h2d + _from_gpu，因为 Tensor 没有 .to()）。"""
    arr = np.ascontiguousarray(np_arr, np.float32)
    if getattr(ref, "device", "cpu") != "cuda":
        return _as_tensor(arr)
    rt = _cuda()
    return Tensor._from_gpu(rt.h2d(arr), requires_grad=False)


class Dropout(Module):
    """inverted dropout。`p` 是丢弃概率（与 torch 一致）。"""

    def __init__(self, p=0.5):
        super().__init__()
        p = float(p)
        if not (0.0 <= p < 1.0):
            raise ValueError("Dropout 的 p 必须在 [0, 1) 内，收到 %r" % (p,) +
                             "。修法：p=0 表示不丢弃；p=1 无意义")
        self.p = p

    def forward(self, x):
        if not getattr(self, "training", True) or self.p == 0.0:
            return x
        shape = tuple(int(s) for s in x.shape)
        keep = (np.random.random(shape) >= self.p).astype(np.float32)
        mask = keep * (1.0 / (1.0 - self.p))
        return x * _like(mask, x)


# ---------------- 池化：索引在主机按形状构造 ----------------
def _pool_plan(N, C, H, W, kh, kw, sh, sw, ph, pw):
    """返回 (oh, ow, idx_full, valid_full)：idx_full 是整张量扁平索引，长度 N*C*OH*OW*kh*kw。"""
    oh = (H + 2 * ph - kh) // sh + 1
    ow = (W + 2 * pw - kw) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("池化输出尺寸非正（H=%d W=%d k=%dx%d s=%d,%d p=%d,%d → %dx%d）"
                         % (H, W, kh, kw, sh, sw, ph, pw, oh, ow) +
                         "。修法：减小 kernel 或 padding，或加大输入")
    k = kh * kw
    total = N * C * oh * ow * k
    idx = np.zeros(total, dtype=np.int64)
    valid = np.zeros(total, dtype=bool)
    t = 0
    for n in range(N):
        for c in range(C):
            base = (n * C + c) * (H * W)
            for i in range(oh):
                ih0 = i * sh - ph
                for j in range(ow):
                    iw0 = j * sw - pw
                    for a in range(kh):
                        ih = ih0 + a
                        okh = 0 <= ih < H
                        for b in range(kw):
                            iw = iw0 + b
                            if okh and 0 <= iw < W:
                                idx[t] = base + ih * W + iw
                                valid[t] = True
                            t += 1
    return oh, ow, idx, valid


def _pool2d(x, kernel, stride, padding, mode):
    x = _as_tensor(x)
    if len(x.shape) != 4:
        raise ValueError("池化期望 NCHW，收到 shape=%s" % (tuple(x.shape),) +
                         "。修法：图像数据用 [N,C,H,W]；序列数据用 pool1d 或先 reshape")
    N, C, H, W = (int(s) for s in x.shape)
    kh, kw = _pair(kernel)
    sh, sw = _pair(stride if stride is not None else kernel)
    ph, pw = _pair(padding)
    oh, ow, idx, valid = _pool_plan(N, C, H, W, kh, kw, sh, sw, ph, pw)
    k = kh * kw
    P = N * C * oh * ow
    if mode == "avg":
        win = gather_flat(x, idx, n_src=N * C * H * W)
        v = _like(valid.astype(np.float32), x)
        win = win * v
        y = win.reshape(P, k).mean(axis=1)          # 等价 count_include_pad=True
    elif mode == "max":
        xn = np.asarray(x.to_numpy(), np.float32).reshape(-1)
        vals = np.where(valid, xn[idx], -np.inf).reshape(P, k)
        pick = vals.argmax(axis=1)
        best = idx.reshape(P, k)[np.arange(P), pick]
        y = gather_flat(x, best, n_src=N * C * H * W)
    else:
        raise ValueError("未知池化模式 %r" % (mode,) + "。修法：用 max 或 avg")
    return y.reshape(N, C, oh, ow)


def _pool1d(x, kernel, stride, padding, mode):
    x = _as_tensor(x)
    if len(x.shape) != 3:
        raise ValueError("pool1d 期望 [N,C,L]，收到 shape=%s" % (tuple(x.shape),) +
                         "。修法：把通道维放在第 1 维")
    N, C, L = (int(s) for s in x.shape)
    y = _pool2d(x.reshape(N, C, 1, L), (1, _pair(kernel)[0]),
                (1, _pair(stride if stride is not None else kernel)[0]),
                (0, _pair(padding)[0]), mode)
    return y.reshape(N, C, int(y.shape[3]))


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, x):
        return _pool2d(x, self.kernel_size, self.stride, self.padding, "max")


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, x):
        return _pool2d(x, self.kernel_size, self.stride, self.padding, "avg")


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, x):
        return _pool1d(x, self.kernel_size, self.stride, self.padding, "max")


class AvgPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, x):
        return _pool1d(x, self.kernel_size, self.stride, self.padding, "avg")


# ---------------- BatchNorm ----------------
class _BatchNorm(Module):
    """两遍法 BatchNorm：均值 → 中心化后求方差；训练时更新 running 统计。"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        if affine:
            self.weight = Parameter(np.ones(self.num_features, np.float32))
            self.bias = Parameter(np.zeros(self.num_features, np.float32))
        else:
            self.weight = None
            self.bias = None
        self.running_mean = np.zeros(self.num_features, np.float32)
        self.running_var = np.ones(self.num_features, np.float32)

    def _reduce(self, x, reduce_axes):
        """沿 reduce_axes 依次做单轴 mean（CUDA 的 mean 只支持单轴）。"""
        y = x
        for ax in sorted(reduce_axes, reverse=True):
            y = y.mean(axis=ax)
        return y

    def forward(self, x):
        x = _as_tensor(x)
        shape = tuple(int(s) for s in x.shape)
        if len(shape) < 2:
            raise ValueError("BatchNorm 期望至少 [N,C,...]，收到 shape=%s" % (shape,) +
                             "。修法：检查输入维度")
        C = shape[1]
        if C != self.num_features:
            raise ValueError("BatchNorm 的 num_features=%d 与输入通道 %d 不一致"
                             % (self.num_features, C) + "。修法：按输入通道数建 BatchNorm")
        reduce_axes = [0] + list(range(2, len(shape)))
        if getattr(self, "training", True):
            m = self._reduce(x, reduce_axes)                     # [C]
            m3 = m.reshape(*([1, C] + [1] * (len(shape) - 2)))
            d = x - m3
            v = self._reduce(d * d, reduce_axes)                 # [C]（总体方差）
            mn = np.asarray(m.to_numpy(), np.float32).reshape(-1)
            vn = np.asarray(v.to_numpy(), np.float32).reshape(-1)
            n = 1
            for ax in reduce_axes:
                n *= shape[ax]
            mom = self.momentum
            self.running_mean = (1 - mom) * self.running_mean + mom * mn
            unb = vn * (n / max(n - 1, 1))
            self.running_var = (1 - mom) * self.running_var + mom * unb
        else:
            m3 = _like(self.running_mean.reshape(*([1, C] + [1] * (len(shape) - 2))), x)
            d = x - m3
            v = _like(self.running_var.reshape(-1), x)
        v3 = v.reshape(*([1, C] + [1] * (len(shape) - 2)))
        y = d / _sqrt(v3 + self.eps)
        if self.affine:
            y = y * self.weight.reshape(*([1, C] + [1] * (len(shape) - 2)))
            y = y + self.bias.reshape(*([1, C] + [1] * (len(shape) - 2)))
        return y


def _sqrt(t):
    from . import sqrt as _s
    return _s(t)


class BatchNorm1d(_BatchNorm):
    pass


class BatchNorm2d(_BatchNorm):
    pass
