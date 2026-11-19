# -*- coding: utf-8 -*-
"""LayerNorm（与 torch.nn.LayerNorm 对齐）。

实现方式：**纯组合已有可微算子**，不新写内核：

    mu  = mean(x, -1)
    xc  = x - mu
    var = mean(xc * xc, -1)
    y   = xc * (1 / sqrt(var + eps)) * gamma + beta

因此只要 `mean(axis, keepdims=True)` / `sqrt` / 广播在设备上可用，它在 CPU 与 CUDA 上都能跑
（是否真的能在 CUDA 跑，由测试实测，不靠推断）。
"""
import numpy as np

from . import Tensor, _as_tensor, nn

__all__ = ["LayerNorm"]


class LayerNorm(object):
    """归一化最后一维（torch 语义：eps 加在方差上，含可学习的 gamma/beta）。"""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        dim = int(normalized_shape) if not isinstance(normalized_shape, (tuple, list)) \
            else int(np.prod(normalized_shape))
        self.normalized_shape = (dim,)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(np.ones(dim, np.float32))
            self.bias = nn.Parameter(np.zeros(dim, np.float32))
        else:
            self.weight = None
            self.bias = None

    def parameters(self):
        return [p for p in (self.weight, self.bias) if p is not None]

    def __call__(self, x):
        x = _as_tensor(x)
        if not len(x.shape):
            raise ValueError("LayerNorm 需要至少 1 维输入。修法：传 [..., dim] 形状的张量")
        dim = int(x.shape[-1])
        if dim != self.normalized_shape[0]:
            raise ValueError("LayerNorm 维度不匹配：权重是 %d，输入最后一维是 %d。修法：检查模型配置"
                             % (self.normalized_shape[0], dim))
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        inv = 1.0 / (var + self.eps).sqrt()
        y = xc * inv
        if self.elementwise_affine:
            y = y * self.weight + self.bias
        return y
