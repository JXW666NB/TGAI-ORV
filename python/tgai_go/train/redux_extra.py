# -*- coding: utf-8 -*-
"""补齐 PyTorch 的归约算子：`max` / `min` / `var` / `std` / `norm`（Tensor 方法）。

## 实现与代价（如实说明）

* **max / min**：本实现先用 argmax/argmin 得到位置，构造 **one-hot mask**，再算
  `(x * mask).sum(axis)`。好处是梯度能自动路由（只有 argmax 位置的梯度非零），
  不需要新内核；代价是 **CUDA 上要一次 `to_numpy()`**（取 argmax 需要值）和一次 mask 的
  主机→设备拷贝。大张量、频繁调用时明显慢于 cuDNN 原生归约。
  并列最大值取**第一个**（与 torch CPU 行为一致；torch CUDA 在并列时不保证）。
* **var / std**：两遍法（先均值、再中心化后平方求均值），只用已有 `mean/mul/add`，
  CPU/CUDA 通用；`unbiased=True` 时乘 `n/(n-1)`（与 torch 默认一致）。
* **norm**：只支持 `p=2`（`sqrt(sum(x*x))`），其他 p 明确报错。
"""
import numpy as np

from . import Tensor, _as_tensor, _cuda

__all__ = ["max", "min", "var", "std", "norm"]


def _host_values(t):
    if getattr(t, "device", "cpu") == "cuda":
        return np.asarray(t.to_numpy(), np.float32)
    return np.asarray(t.data, np.float32)


def _like(np_arr, ref):
    arr = np.ascontiguousarray(np_arr, np.float32)
    if getattr(ref, "device", "cpu") != "cuda":
        return _as_tensor(arr)
    return Tensor._from_gpu(_cuda().h2d(arr), requires_grad=False)


def _norm_axis(axis, ndim):
    if axis is None:
        return None
    ax = int(axis)
    if ax < 0:
        ax += ndim
    if ax < 0 or ax >= ndim:
        raise ValueError("归约轴 %r 越界（ndim=%d）" % (axis, ndim) +
                         "。修法：检查 dim/axis 取值（负数按从后往前数）")
    return ax


def _rowwise_max_cuda(x, keepdims):
    """末维 max：走引擎内核（一块一行 + warp 归约），不再 D2H + one-hot mask。

    语义与 `_masked_reduce` 逐位等价（并列取第一个），但每次调用从
    "1 次强制同步 + 全量 D2H + 全量 H2D" 变成 "1 次内核发射"。
    块化注意力每对块要调 2 次，这条是它慢以及 CUDA Graph 无法捕获的根因。
    """
    rt = _cuda()
    shape = tuple(int(s) for s in x.shape)
    k = shape[-1]
    nrows = int(x.numel()) // k if k else 0
    val, idx = rt.rowwise_max(x._gpu)
    out_shape = shape[:-1] + ((1,) if keepdims else ())
    val_buf = rt.reshape(val, out_shape) if int(np.prod(out_shape)) == nrows else val
    need = bool(x.requires_grad)
    out = Tensor._from_gpu(val_buf, requires_grad=need, _parents=(x,) if need else ())
    if not need:
        return out

    def bw(g, grads):
        gflat = rt.reshape(g, (nrows,))
        dx = rt.rowwise_max_bwd(gflat, idx, x.shape)
        _acc_local(grads, x, dx)

    out._backward = bw
    return out


def _acc_local(grads, node, g):
    from . import _acc as _real_acc
    _real_acc(grads, node, g)


def _masked_reduce(self, axis, keepdims, pick):
    """用 one-hot mask 实现 max/min，使梯度只落到被选中的位置。"""
    self = _as_tensor(self)
    shape = tuple(int(s) for s in self.shape)
    ax = _norm_axis(axis, len(shape))
    if (getattr(self, "device", "cpu") == "cuda" and pick is np.argmax
            and ax is not None and ax == len(shape) - 1 and len(shape) >= 2):
        return _rowwise_max_cuda(self, keepdims)
    xn = _host_values(self)
    mask = np.zeros_like(xn)
    if ax is None:
        flat = int(pick(xn))
        mask.reshape(-1)[flat] = 1.0
    else:
        arg = xn.argmax(axis=ax) if pick is np.argmax else xn.argmin(axis=ax)
        np.put_along_axis(mask, np.expand_dims(arg, ax), 1.0, axis=ax)
    y = (self * _like(mask, self)).sum(axis=ax, keepdims=bool(keepdims))
    return y


def max(self, axis=None, keepdims=False, dim=None, keepdim=None):
    """对应 `torch.max(x, dim)` / `x.max()`。返回归约结果（不返回 (values, indices)）。"""
    if dim is not None:
        axis = dim
    if keepdim is not None:
        keepdims = keepdim
    return _masked_reduce(self, axis, keepdims, np.argmax)


def min(self, axis=None, keepdims=False, dim=None, keepdim=None):
    """对应 `torch.min(x, dim)` / `x.min()`。"""
    if dim is not None:
        axis = dim
    if keepdim is not None:
        keepdims = keepdim
    return _masked_reduce(self, axis, keepdims, np.argmin)


def _reduced_count(shape, ax):
    if ax is None:
        n = 1
        for s in shape:
            n *= int(s)
        return n
    return int(shape[ax])


def var(self, axis=None, keepdims=False, unbiased=True, dim=None, keepdim=None):
    """方差。`unbiased=True`（默认）与 torch 一致（除以 n-1）。"""
    if dim is not None:
        axis = dim
    if keepdim is not None:
        keepdims = keepdim
    self = _as_tensor(self)
    ax = _norm_axis(axis, len(tuple(self.shape)))
    m = self.mean(axis=ax, keepdims=True)
    d = self - m
    v = (d * d).mean(axis=ax, keepdims=bool(keepdims))
    if unbiased:
        n = _reduced_count(tuple(int(s) for s in self.shape), ax)
        if n > 1:
            v = v * (float(n) / float(n - 1))
    return v


def std(self, axis=None, keepdims=False, unbiased=True, dim=None, keepdim=None):
    from . import sqrt as _sqrt
    return _sqrt(var(self, axis=axis, keepdims=keepdims, unbiased=unbiased,
                     dim=dim, keepdim=keepdim))


def norm(self, p=2, axis=None, keepdims=False, dim=None, keepdim=None):
    """只支持 p=2：`sqrt(sum(x^2))`。"""
    if dim is not None:
        axis = dim
    if keepdim is not None:
        keepdims = keepdim
    if p not in (2, 2.0, "fro"):
        raise NotImplementedError("norm 只支持 p=2（当前 p=%r）" % (p,) +
                                  "。修法：用 (x.abs()**p).sum() 组合，或告知需要哪种范数")
    from . import sqrt as _sqrt
    self = _as_tensor(self)
    ax = _norm_axis(axis, len(tuple(self.shape)))
    return _sqrt((self * self).sum(axis=ax, keepdims=bool(keepdims)))
