# -*- coding: utf-8 -*-
"""contrib · 激活与逐元素算子（PyTorch 同名同义，均带解析反向）。

为什么这些不进核心：LLM 训练里用不到（核心只需要 silu / gelu(tanh) / softmax / rmsnorm）。
但别人从 PyTorch 迁过来的脚本可能用到，所以放官方 contrib。
"""
import numpy as np

from ._base import numpy_op

F32 = np.float32


def _erf(x):
    """向量化 erf：用 math.erf，避免依赖 scipy。"""
    import math
    return np.vectorize(math.erf, otypes=[x.dtype])(x)


# ---------------------------------------------------------------- erf / gelu
@numpy_op("erf", doc="误差函数 erf(x)；精确 GELU 与统计类损失要用")
def _():
    def fwd(x):
        return _erf(x.astype(np.float64)).astype(F32)

    def bwd(g, out, xs):
        (x,) = xs
        return [(g * (2.0 / np.sqrt(np.pi)) * np.exp(-x.astype(np.float64) ** 2)).astype(F32)]

    return fwd, bwd


@numpy_op("gelu_exact", doc="精确 GELU（erf 版），核心只内置 tanh 近似")
def _():
    from ..contrib.activations import _erf

    def fwd(x):
        xd = x.astype(np.float64)
        return (xd * 0.5 * (1.0 + _erf(xd / np.sqrt(2.0)))).astype(F32)

    def bwd(g, out, xs):
        (x,) = xs
        xd = x.astype(np.float64)
        cdf = 0.5 * (1.0 + _erf(xd / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * xd * xd) / np.sqrt(2.0 * np.pi)
        return [(g * (cdf + xd * pdf)).astype(F32)]

    return fwd, bwd


# ------------------------------------------------------------------ ELU / SELU
@numpy_op("elu", doc="ELU，参数 alpha 默认 1.0（与 torch 一致）")
def _():
    def fwd(x, alpha=1.0):
        a = np.asarray(x, dtype=np.float64)
        return np.where(a > 0, a, alpha * np.expm1(np.minimum(a, 0.0))).astype(F32)

    def bwd(g, out, xs, alpha=1.0):
        (x,) = xs
        a = x.astype(np.float64)
        d = np.where(a > 0, 1.0, alpha * np.exp(np.minimum(a, 0.0)))
        return [(g * d).astype(F32)]

    return fwd, bwd


@numpy_op("selu", doc="SELU（固定 alpha=1.6732632423543772, scale=1.0507009873554805）")
def _():
    ALPHA = 1.6732632423543772
    SCALE = 1.0507009873554805

    def fwd(x):
        a = x.astype(np.float64)
        return (SCALE * np.where(a > 0, a, ALPHA * np.expm1(np.minimum(a, 0.0)))).astype(F32)

    def bwd(g, out, xs):
        (x,) = xs
        a = x.astype(np.float64)
        d = SCALE * np.where(a > 0, 1.0, ALPHA * np.exp(np.minimum(a, 0.0)))
        return [(g * d).astype(F32)]

    return fwd, bwd


@numpy_op("leaky_relu", doc="LeakyReLU，negative_slope 默认 0.01")
def _():
    def fwd(x, negative_slope=0.01):
        a = x.astype(np.float64)
        return np.where(a > 0, a, a * negative_slope).astype(F32)

    def bwd(g, out, xs, negative_slope=0.01):
        (x,) = xs
        a = x.astype(np.float64)
        return [(g * np.where(a > 0, 1.0, negative_slope)).astype(F32)]

    return fwd, bwd


# --------------------------------------------------------- softplus / mish
@numpy_op("softplus", doc="Softplus，数值稳定写法；beta 默认 1.0")
def _():
    def fwd(x, beta=1.0, threshold=20.0):
        a = x.astype(np.float64) * beta
        out = np.where(a > threshold, x.astype(np.float64),
                       np.log1p(np.exp(-np.abs(a))) + np.maximum(a, 0.0) / beta)
        return out.astype(F32)

    def bwd(g, out, xs, beta=1.0, threshold=20.0):
        (x,) = xs
        a = x.astype(np.float64) * beta
        s = 1.0 / (1.0 + np.exp(-np.clip(a, -60, 60)))
        return [(g * s).astype(F32)]

    return fwd, bwd


@numpy_op("mish", doc="Mish: x * tanh(softplus(x))")
def _():
    def _sp(a):
        return np.log1p(np.exp(-np.abs(a))) + np.maximum(a, 0.0)

    def fwd(x):
        a = x.astype(np.float64)
        return (a * np.tanh(_sp(a))).astype(F32)

    def bwd(g, out, xs):
        (x,) = xs
        a = x.astype(np.float64)
        sp = _sp(a)
        th = np.tanh(sp)
        sig = 1.0 / (1.0 + np.exp(-np.clip(a, -60, 60)))
        return [(g * (th + a * (1.0 - th * th) * sig)).astype(F32)]

    return fwd, bwd


# ------------------------------------------------- hard swish / hard sigmoid
@numpy_op("hardswish", doc="HardSwish: x * clamp(x+3,0,6)/6")
def _():
    def fwd(x):
        a = x.astype(np.float64)
        return (a * np.clip(a + 3.0, 0.0, 6.0) / 6.0).astype(F32)

    def bwd(g, out, xs):
        (x,) = xs
        a = x.astype(np.float64)
        c = np.clip(a + 3.0, 0.0, 6.0)
        inside = ((a > -3.0) & (a < 3.0)).astype(np.float64)
        return [(g * (c / 6.0 + a * inside / 6.0)).astype(F32)]

    return fwd, bwd


@numpy_op("hardsigmoid", doc="HardSigmoid: clamp(x+3,0,6)/6")
def _():
    def fwd(x):
        return (np.clip(x.astype(np.float64) + 3.0, 0.0, 6.0) / 6.0).astype(F32)

    def bwd(g, out, xs):
        (x,) = xs
        inside = ((x > -3.0) & (x < 3.0)).astype(np.float64)
        return [(g * inside / 6.0).astype(F32)]

    return fwd, bwd


# ------------------------------------------------------------ 简单逐元素
@numpy_op("log1p", doc="log(1+x)，小 x 时比 log(1+x) 精确")
def _():
    def fwd(x):
        return np.log1p(x.astype(np.float64)).astype(F32)

    def bwd(g, out, xs):
        (x,) = xs
        return [(g / (1.0 + x.astype(np.float64))).astype(F32)]

    return fwd, bwd


@numpy_op("sign", doc="符号函数（反向为 0）")
def _():
    def fwd(x):
        return np.sign(x).astype(F32)

    def bwd(g, out, xs):
        return [np.zeros_like(xs[0], dtype=F32)]

    return fwd, bwd


@numpy_op("floor", doc="向下取整（反向为 0）")
def _():
    def fwd(x):
        return np.floor(x).astype(F32)

    def bwd(g, out, xs):
        return [np.zeros_like(xs[0], dtype=F32)]

    return fwd, bwd


@numpy_op("ceil", doc="向上取整（反向为 0）")
def _():
    def fwd(x):
        return np.ceil(x).astype(F32)

    def bwd(g, out, xs):
        return [np.zeros_like(xs[0], dtype=F32)]

    return fwd, bwd


@numpy_op("trunc", doc="截断取整（反向为 0）")
def _():
    def fwd(x):
        return np.trunc(x).astype(F32)

    def bwd(g, out, xs):
        return [np.zeros_like(xs[0], dtype=F32)]

    return fwd, bwd


@numpy_op("lerp", doc="lerp(a,b,w) = a + w*(b-a)，与 torch.lerp 一致")
def _():
    def fwd(a, b, w):
        aa, bb = a.astype(np.float64), b.astype(np.float64)
        ww = np.asarray(w, dtype=np.float64)
        return (aa + ww * (bb - aa)).astype(F32)

    def bwd(g, out, xs):
        a, b, w = xs
        aa, bb = a.astype(np.float64), b.astype(np.float64)
        ww = np.asarray(w, dtype=np.float64)
        return [(g * (1.0 - ww)).astype(F32), (g * ww).astype(F32),
                (g * (bb - aa)).astype(F32)]

    return fwd, bwd
