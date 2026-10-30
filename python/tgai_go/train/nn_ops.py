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

from . import Tensor, _acc
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


# ===========================================================================
# 第二批核心算子（推理 + 训练都用得到）
#   归约：argmax / amax / amin / norm
#   损失：nll_loss / mse_loss
#   形状：chunk / tril / arange
#   采样：topk_indices（推理解码用）
#   优化：sgd_step
# 说明：这里用「numpy 前向 + 解析反向」实现（与 ops_plugin 同机制），
#       每个都在 tests/test_nn_ops.py 里与 PyTorch 对齐。
# ===========================================================================


def _numpy_op(name, fwd, bwd):
    """把 (numpy 前向, 解析反向) 包装成带 autograd 的算子（不注册到插件表）。"""

    def op(*args, **kw):
        tensors = [a for a in args if isinstance(a, Tensor)]
        arrs = [a.data if isinstance(a, Tensor) else a for a in args]
        out_data = np.asarray(fwd(*arrs, **kw), dtype=np.float32)
        need = any(getattr(a, "requires_grad", False) for a in args)
        out = Tensor(out_data, requires_grad=need, _parents=tuple(tensors))

        def bw(g, grads):
            gs = bwd(np.asarray(g, np.float32), out_data, arrs, **kw)
            if not isinstance(gs, (list, tuple)):
                gs = [gs]
            for a, gi in zip(args, gs):
                if isinstance(a, Tensor) and a.requires_grad and gi is not None:
                    _acc(grads, a, np.asarray(gi, dtype=np.float32))

        out._backward = bw
        return out

    op.__name__ = name
    op.__doc__ = "见 nn_ops.%s（与 torch 同名同义）" % (name,)
    return op


argmax = _numpy_op(
    "argmax",
    fwd=lambda x, dim=None: (np.argmax(x, axis=dim) if dim is not None else np.argmax(x)),
    bwd=lambda g, out, xs, dim=None: [np.zeros_like(xs[0], dtype=np.float32)],
)


def _ext_reduce(fn, tag):
    def fwd(x, dim=None, keepdim=False):
        if dim is None:
            return np.asarray(fn(x), dtype=np.float32)
        return np.asarray(fn(x, axis=dim, keepdims=keepdim), dtype=np.float32)

    def bwd(g, out, xs, dim=None, keepdim=False):
        # 与 torch 一致：最大值有并列时，梯度在并列位置**均分**（不是每个都给全额）
        x = xs[0].astype(np.float64)
        scale = float(np.asarray(g).reshape(-1)[0]) if np.ndim(g) == 0 else None
        if dim is None:
            target = np.asarray(fn(x))
            mask = (x == target).astype(np.float64)
            cnt = mask.sum()
            gg = scale if scale is not None else float(np.asarray(g).sum())
            return [((gg / cnt) * mask).astype(np.float32)]
        out_exp = np.expand_dims(out, dim) if not keepdim else out
        mask = (x == out_exp).astype(np.float64)
        cnt = mask.sum(axis=dim, keepdims=True)
        g_exp = (np.expand_dims(g, dim) if not keepdim else g).astype(np.float64)
        return [((g_exp / cnt) * mask).astype(np.float32)]

    return _numpy_op(tag, fwd=fwd, bwd=bwd)


amax = _ext_reduce(np.max, "amax")
amin = _ext_reduce(np.min, "amin")


def norm(x, p=2.0, dim=None, keepdim=False, eps=1e-12):
    """L2 范数（p=2）。梯度裁剪、评估指标都要用。"""
    if float(p) != 2.0:
        raise ValueError("norm: 目前只支持 p=2（收到 p=%r）。修法：用 p=2，或先 sqrt((x**2).sum())" % (p,))
    sq = x * x
    # 注意：Tensor.sum 的关键字是 keepdims（不是 keepdim）
    s = sq.sum(dim, keepdims=keepdim) if dim is not None else sq.sum()
    return (s + eps).sqrt()


def mse_loss(a, b, reduction="mean"):
    """均方误差；reduction 支持 'mean' / 'sum' / 'none'（none 时仍返回已折叠前的张量）。"""
    d = a - b
    sq = d * d
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    if reduction == "none":
        return sq
    raise ValueError("mse_loss: 不支持的 reduction=%r。修法：用 'mean'/'sum'/'none'" % (reduction,))


def nll_loss(log_probs, target, reduction="mean"):
    """负对数似然：log_probs[M,V] 与 target[M]（整数）。

    与 `torch.nn.functional.nll_loss` 一致（target 越界会被明确报错，而不是静默出错）。
    """
    lp = np.asarray(log_probs.to_numpy(), dtype=np.float64)
    tg = np.asarray(target, dtype=np.int64).reshape(-1)
    if lp.ndim != 2:
        raise ValueError("nll_loss: log_probs 必须是 [N, V]，实际 %s。修法：先 log_softmax 再 reshape"
                         % (lp.shape,))
    if lp.shape[0] != tg.shape[0]:
        raise ValueError(
            "nll_loss: 样本数不一致 —— log_probs 有 %d 行，target 有 %d 个。修法：检查 batch 维度"
            % (lp.shape[0], tg.shape[0]))
    if tg.size and (tg.min() < 0 or tg.max() >= lp.shape[1]):
        raise ValueError(
            "nll_loss: target 越界 —— 取值 [%d, %d]，词表大小 %d。修法：检查标签是否含 ignore_index 之外的值"
            % (int(tg.min()), int(tg.max()), lp.shape[1]))
    n = lp.shape[0]
    picked = lp[np.arange(n), tg]
    if reduction == "mean":
        val = -picked.mean()
    elif reduction == "sum":
        val = -picked.sum()
    else:
        raise ValueError("nll_loss: 不支持的 reduction=%r。修法：用 'mean'/'sum'" % (reduction,))

    out = Tensor(np.asarray(val, np.float32), requires_grad=log_probs.requires_grad,
                 _parents=(log_probs,))

    def bw(g, grads):
        if not log_probs.requires_grad:
            return
        scale = float(np.asarray(g).reshape(-1)[0])
        dlp = np.zeros_like(lp)
        coef = -1.0 / n if reduction == "mean" else -1.0
        dlp[np.arange(n), tg] = coef
        _acc(grads, log_probs, (dlp * scale).astype(np.float32))

    out._backward = bw
    return out


def chunk(x, chunks, dim=0):
    """沿 dim 等分成 chunks 份（与 torch.chunk 一致：除不尽时最后一份更小）。"""
    if chunks <= 0:
        raise ValueError("chunk: chunks 必须 > 0，收到 %r" % (chunks,))
    n = x.shape[dim]
    size = int(np.ceil(n / chunks))
    return split(x, size, dim)


def tril(x, diagonal=0):
    """下三角（其余置 0）。与 torch.tril 一致。"""
    n, m = x.shape[-1], x.shape[-2] if len(x.shape) > 1 else x.shape[-1]
    rows, cols = np.tril_indices(m, k=diagonal)
    mask = np.zeros((m, n), dtype=np.float32)
    mask[rows, cols] = 1.0
    return x * _tensor(mask)


def arange(start, end=None, step=1, dtype=np.float32):
    """与 torch.arange 一致的区间张量（只读，不参与求导）。"""
    if end is None:
        start, end = 0, start
    if step == 0:
        raise ValueError("arange: step 不能为 0")
    return _tensor(np.arange(start, end, step, dtype=np.float32))


def topk_indices(x, k, dim=-1, largest=True):
    """返回 top-k 的**索引**（numpy int64）。推理解码（top-k 采样）要用。

    只返回索引、不带梯度 —— 采样是离散决策，本来就不该有梯度。
    """
    arr = np.asarray(x.to_numpy())
    kk = int(k)
    if not (1 <= kk <= arr.shape[dim]):
        raise ValueError(
            "topk_indices: k=%d 超出范围 [1, %d]。修法：k 不得超过该维长度"
            % (kk, arr.shape[dim]))
    if largest:
        idx = np.argpartition(-arr, kk - 1, axis=dim)[
            tuple(slice(None) if i != dim % arr.ndim else slice(0, kk)
                  for i in range(arr.ndim))]
    else:
        idx = np.argpartition(arr, kk - 1, axis=dim)[
            tuple(slice(None) if i != dim % arr.ndim else slice(0, kk)
                  for i in range(arr.ndim))]
    return np.asarray(idx, dtype=np.int64)


def sgd_step(params, lr, momentum=0.0, weight_decay=0.0, dampening=0.0, state=None):
    """SGD（可选 momentum / weight_decay），与 torch.optim.SGD 的更新式一致。

    state: dict，跨步保存动量缓冲（传同一个对象进来即可）。
    返回 state。
    """
    if state is None:
        state = {"momentum_buffers": {}}
    bufs = state.setdefault("momentum_buffers", {})
    for i, p in enumerate(params):
        g = getattr(p, "grad", None)
        if g is None:
            continue
        gd = np.asarray(g, dtype=np.float32)
        if weight_decay:
            gd = gd + np.float32(weight_decay) * np.asarray(p.data, np.float32)
        if momentum:
            buf = bufs.get(i)
            if buf is None:
                bufs[i] = gd.copy()
            else:
                bufs[i] = np.float32(momentum) * buf + np.float32(1.0 - dampening) * gd
            gd = bufs[i]
        p.data[...] = np.asarray(p.data, np.float32) - np.float32(lr) * gd
    return state
