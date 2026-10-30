# -*- coding: utf-8 -*-
"""PyTorch 兼容算子层（第一批）。

为什么单独一个模块而不是改 `__init__.py`：
  * 不碰核心文件，回归风险最小（改核心必须跑 GPU doctor，而本机没有 GPU）；
  * 每个算子都**只由已有 Tensor 算子组合**，所以 CUDA/CPU 都能用；
  * 名字、语义、参数顺序尽量与 PyTorch 一致，方便直接 `from tgai_go.train import nn_ops`。

约定：与 `torch.nn.functional` 同名同义（`x @ W.T + b` 这种细节严格对齐）。
每个函数都在 `tests/test_nn_ops.py` 里与 PyTorch 做数值对齐。

已知边界（写在这里，不藏）：
  * `embedding` / `gather` 目前用 numpy 查表实现，**不支持反向**（没有 gather 内核）。
    这是 Tier-1 待攻破项，训练要用必须补内核。
  * `gelu` 只实现 tanh 近似（精确版需要 erf，我们还没有）。
"""
import numpy as np

from . import Tensor
from . import tensor as _tensor


def _ones_like(x):
    return x * 0.0 + 1.0


def rsqrt(x):
    """1/sqrt(x)，与 torch.rsqrt 一致（负数会产生 nan，不带断言）。"""
    return _ones_like(x) / x.sqrt()


def pow(x, p):
    """幂。p 为整数时用连乘（保号正确）；非整数只对 x>0 有定义，用 exp/log 实现。"""
    if isinstance(p, (int, float)) and float(p).is_integer():
        n = int(abs(p))
        out = _ones_like(x)
        base = x if p >= 0 else _ones_like(x) / x
        for _ in range(n):
            out = out * base
        return out
    return (x.log() * float(p)).exp()


def sigmoid(x):
    """1/(1+exp(-x))。"""
    return _ones_like(x) / (_ones_like(x) + (-x).exp())


def tanh(x):
    """tanh(x) = 2*sigmoid(2x) - 1。"""
    return sigmoid(x * 2.0) * 2.0 - 1.0


def gelu(x, approximate="tanh"):
    """GELU。目前只支持 tanh 近似（精确版需要 erf）。"""
    if approximate not in ("tanh", None, "none"):
        raise ValueError("gelu: 只支持 approximate='tanh'（精确版需要 erf 算子，尚未实现）")
    c = float(np.sqrt(2.0 / np.pi))
    inner = (x + (x * x * x) * 0.044715) * c
    return x * 0.5 * (_ones_like(x) + tanh(inner))


def linear(x, weight, bias=None):
    """y = x @ weight.T + bias —— 与 torch.nn.functional.linear 一致（weight 是 [out, in]）。"""
    if weight.ndim != 2:
        raise ValueError("linear: weight 必须是 2 维 [out, in]，实际 %s" % (weight.shape,))
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            "linear: 形状不匹配 —— x 最后一维 %d，weight 的 in_features %d。"
            "修法：weight 应为 [out, in]，或把 x 转成 last-dim=in_features"
            % (x.shape[-1], weight.shape[1]))
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def rms_norm(x, weight, eps=1e-6):
    """x / sqrt(mean(x^2)+eps) * weight —— 与 torch.nn.functional.rms_norm 一致。"""
    s = (x * x).mean(-1, keepdim=True)
    d = (s + eps).sqrt()
    return (x / d) * weight


def layer_norm(x, weight=None, bias=None, eps=1e-5):
    """(x-mean)/sqrt(var+eps)*w + b，var 用有偏估计（与 torch 默认一致）。"""
    mu = x.mean(-1, keepdim=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdim=True)
    out = xc / (var + eps).sqrt()
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def embedding(weight, ids):
    """查表。⚠ 目前走 numpy，**不支持反向**（Tier-1 待补 gather 内核）。"""
    w = np.asarray(weight.to_numpy(), dtype=np.float32)
    idx = np.asarray(ids, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= w.shape[0]):
        raise ValueError(
            "embedding: 索引越界 —— ids 范围 [%d, %d]，词表大小 %d。"
            "修法：检查 tokenizer 的 vocab 上限与 special token id"
            % (int(idx.min()), int(idx.max()), w.shape[0]))
    return _tensor(w[idx])


def stack(xs, dim=0):
    """沿新维度拼接（与 torch.stack 一致）。"""
    xs = list(xs)
    if not xs:
        raise ValueError("stack: 输入为空")
    shp = xs[0].shape
    for x in xs:
        if x.shape != shp:
            raise ValueError("stack: 形状不一致 %s vs %s。修法：先用 reshape 对齐再 stack"
                             % (shp, x.shape))
    nd = len(shp)
    if dim < 0:
        dim += nd + 1
    parts = []
    for x in xs:
        newshape = list(shp[:dim]) + [1] + list(shp[dim:])
        parts.append(x.reshape(*newshape))
    from . import cat
    return cat(parts, dim=dim)


def split(x, sizes, dim=0):
    """按尺寸切分，返回 list（与 torch.split 一致；sizes 可为 int 或 list）。"""
    n = x.shape[dim]
    if isinstance(sizes, int):
        sizes = [sizes] * (n // sizes) + ([n % sizes] if n % sizes else [])
    out, start = [], 0
    for s in sizes:
        idx = [slice(None)] * len(x.shape)
        idx[dim] = slice(start, start + s)
        out.append(x[tuple(idx)])
        start += s
    return out


def triu(x, diagonal=0):
    """上三角（其余置 0）。用掩码乘法实现，保 autograd。"""
    n, m = x.shape[-1], x.shape[-2] if len(x.shape) > 1 else x.shape[-1]
    rows, cols = np.triu_indices(m, k=diagonal)
    mask = np.zeros((m, n), dtype=np.float32)
    mask[rows, cols] = 1.0
    return x * _tensor(mask)


def where(cond, a, b):
    """torch.where(cond, a, b)。cond 需为 0/1 张量（我们还没有 bool dtype）。"""
    m = cond
    return a * m + b * (_ones_like(m) - m)


def clip_grad_norm_(params, max_norm):
    """梯度 L2 范数裁剪（与 torch.nn.utils.clip_grad_norm_ 一致），返回裁剪前总范数。"""
    total = 0.0
    for p in params:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        arr = np.asarray(g.to_numpy(), dtype=np.float64)
        total += float((arr * arr).sum())
    norm = float(np.sqrt(total))
    if max_norm is not None and max_norm > 0 and norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for p in params:
            g = getattr(p, "grad", None)
            if g is None:
                continue
            p.grad = g * scale
    return norm
