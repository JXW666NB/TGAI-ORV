# -*- coding: utf-8 -*-
"""contrib · 形状/索引类算子（PyTorch 同名同义，反向是它的逆操作）。

这些不进核心：核心只需要 reshape/permute/cat/repeat_interleave 够训练用。
但迁移脚本里常见 flip / roll / flatten / unsqueeze，放官方 contrib。
"""
import numpy as np

from ._base import numpy_op

F32 = np.float32


def _norm_dim(dim, ndim):
    d = int(dim)
    if d < 0:
        d += ndim
    if not (0 <= d < ndim):
        raise ValueError(
            "维度越界：dim=%d，张量只有 %d 维。修法：dim 取 [-%d, %d]"
            % (int(dim), ndim, ndim, ndim - 1))
    return d


@numpy_op("flip", doc="沿给定维度翻转；反向是同一个 flip")
def _():
    def fwd(x, dims):
        return np.flip(x, axis=tuple(dims)).astype(F32)

    def bwd(g, out, xs, **kw):
        dims = xs[1]
        return [np.flip(g, axis=tuple(dims)).astype(F32)]

    return fwd, bwd


@numpy_op("roll", doc="循环移位；反向是 -shift")
def _():
    def fwd(x, shifts, dims=None):
        return np.roll(x, shift=shifts, axis=dims).astype(F32)

    def bwd(g, out, xs, **kw):
        shifts = xs[1]
        dims = xs[2] if len(xs) > 2 else None
        inv = [-v for v in shifts] if isinstance(shifts, (list, tuple)) else -shifts
        return [np.roll(g, shift=inv, axis=dims).astype(F32)]

    return fwd, bwd


@numpy_op("flatten", doc="展平（start_dim/end_dim 与 torch 一致）")
def _():
    def fwd(x, start_dim=0, end_dim=-1):
        a = np.asarray(x)
        sd = _norm_dim(start_dim, a.ndim)
        ed = _norm_dim(end_dim, a.ndim)
        newshape = a.shape[:sd] + (-1,) + a.shape[ed + 1:]
        return a.reshape(newshape).astype(F32)

    def bwd(g, out, xs, **kw):
        return [g.reshape(xs[0].shape).astype(F32)]

    return fwd, bwd


@numpy_op("unsqueeze", doc="插入一个长度为 1 的维度")
def _():
    def fwd(x, dim):
        return np.expand_dims(x, axis=int(dim)).astype(F32)

    def bwd(g, out, xs, **kw):
        return [g.reshape(xs[0].shape).astype(F32)]

    return fwd, bwd


@numpy_op("squeeze", doc="去掉长度为 1 的维度（dim=None 去掉全部）")
def _():
    def fwd(x, dim=None):
        a = np.asarray(x)
        if dim is None:
            return np.squeeze(a).astype(F32)
        d = _norm_dim(dim, a.ndim)
        # 与 torch 一致：目标维度长度不是 1 时**原样返回**（numpy 会报错，语义不同）
        if a.shape[d] != 1:
            return np.array(a, dtype=F32, copy=True)
        return np.squeeze(a, axis=d).astype(F32)

    def bwd(g, out, xs, **kw):
        return [g.reshape(xs[0].shape).astype(F32)]

    return fwd, bwd
