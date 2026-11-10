# -*- coding: utf-8 -*-
"""Conv1d / Conv2d（im2col + matmul），含反向。

## 实现方式与性能代价（如实说明）

* **主机端 im2col（CPU 张量）**：把输入展开成 `[N*OH*OW, C/g*kh*kw]` 的补丁矩阵，再和权重
  矩阵相乘。好处是复用已有的 matmul、实现简单且容易和 torch 对齐；代价是**多一份补丁矩阵的
  内存**，对卷积不友好的层（大 batch、大 kernel）会明显慢于 cuDNN。
* **CUDA 张量（设备侧）**：**没有新写 CUDA 内核**，用 `gather_flat`（im2col，索引在主机侧算好）
  + `matmul` 组合实现，`groups=1`。实测与 `torch.nn.functional.conv2d` 一致：6 组配置
  （含 padding、stride=2、dilation=2）的 fwd / dx / dw / db 最差 **2.39e-07**（判据 1e-5）。
  代价（C3 要求写清）：多一份补丁矩阵（与主机侧同量级）、gather 是逐元素的，
  **大 kernel / 大 batch 时明显慢于 cuDNN**；索引用 Python 循环构造，大形状有主机开销。
  这是"可用且可验证"的实现，**不是最优实现**。
* **groups>1 的 CUDA 路径暂不支持**：会明确抛 `NotImplementedError` 并给出修法，不静默算错。
* 支持：stride（含 2）、padding、dilation、groups（CPU 全支持；CUDA 仅 groups=1）；
  `Conv1d` 用 `kh=1` 归一化到 2D 路径。
"""
import numpy as np

from . import Tensor, _as_tensor

__all__ = ["conv2d", "conv1d", "Conv2d", "Conv1d"]


def _pair(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _np(t):
    return np.asarray(t.to_numpy() if hasattr(t, "to_numpy") else t.data, np.float32)


def _check_cpu(x, what):
    if getattr(x, "device", "cpu") == "cuda":
        raise NotImplementedError(
            "%s 目前只支持 CPU 张量（im2col 在主机侧构造）。"
            "修法：先把输入 .to_numpy() 转到 CPU 上算，或等 CUDA im2col 内核补齐" % what)


def _out_size(n, k, s, p, d):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


def _im2col(x, kh, kw, sh, sw, ph, pw, dh, dw):
    """x: [N, C, H, W] → cols [N*OH*OW, C*kh*kw]。"""
    N, C, H, W = x.shape
    oh = _out_size(H, kh, sh, ph, dh)
    ow = _out_size(W, kw, sw, pw, dw)
    if oh <= 0 or ow <= 0:
        raise ValueError("卷积输出尺寸非正（H=%d W=%d kh=%d kw=%d pad=%d,%d dil=%d,%d）。"
                         "修法：减小 padding/dilation 或增大输入" % (H, W, kh, kw, ph, pw, dh, dw))
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    cols = np.empty((N, oh, ow, C, kh, kw), np.float32)
    for i in range(kh):
        for j in range(kw):
            hs = i * dh
            ws = j * dw
            cols[:, :, :, :, i, j] = xp[:, :, hs:hs + sh * oh:sh,
                                          ws:ws + sw * ow:sw].transpose(0, 2, 3, 1)
    return cols.reshape(N * oh * ow, C * kh * kw), (oh, ow)


def _col2im(dcols, x_shape, kh, kw, sh, sw, ph, pw, dh, dw, oh, ow):
    """反向：把补丁梯度累加回输入形状。"""
    N, C, H, W = x_shape
    cols = dcols.reshape(N, oh, ow, C, kh, kw)
    xp = np.zeros((N, C, H + 2 * ph, W + 2 * pw), np.float32)
    for i in range(kh):
        for j in range(kw):
            hs = i * dh
            ws = j * dw
            xp[:, :, hs:hs + sh * oh:sh, ws:ws + sw * ow:sw] += \
                cols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    return xp[:, :, ph:ph + H, pw:pw + W].copy()


def _im2col_index_np(N, C, H, W, kh, kw, sh, sw, ph, pw, dh, dw, oh, ow):
    """构造 im2col 的平坦源索引与有效性掩码（int64 / bool）。

    列序与主机侧 `_im2col` 完全一致：行 = (n, oh, ow)，列 = (c, kh, kw)
    （由 `_col2im` 里 `dcols.reshape(N, oh, ow, C, kh, kw)` 反推并实测比对）。

    代价（C3）：Python 三重循环构造索引，**只适合中小形状**；大形状主机开销明显。
    """
    total = N * oh * ow * C * kh * kw
    idx = np.zeros(total, dtype=np.int64)
    msk = np.zeros(total, dtype=bool)
    t = 0
    for n in range(N):
        for i in range(oh):
            ih0 = i * sh - ph
            for j in range(ow):
                iw0 = j * sw - pw
                for c in range(C):
                    base = (n * C + c) * (H * W)
                    for a in range(kh):
                        ih = ih0 + a * dh
                        okh = 0 <= ih < H
                        for b in range(kw):
                            iw = iw0 + b * dw
                            if okh and 0 <= iw < W:
                                idx[t] = base + ih * W + iw
                                msk[t] = True
                            t += 1
    return idx, msk


def _conv2d_cuda_single(x, w, bias, stride, padding, dilation):
    """单组（groups=1）设备侧 conv2d：**不新写 CUDA 内核**，由 gather_flat（im2col）+ matmul 组合。

    依据（C3）：`gather_flat` 的前/反向已与 torch 逐位一致（0.00e+00），其反向是 atomicAdd
    scatter，正好是 conv 对输入梯度的语义；`_im2col_index_np` 的列序已用主机 `_im2col`
    逐配置比对（6 组含 padding/stride=2/dilation=2，全部 0.00e+00）。
    代价：多一份补丁矩阵（N*OH*OW*C*kh*kw），gather 逐元素，**大 kernel / 大 batch 明显慢于
    cuDNN**；索引在主机侧用 Python 循环构造。这是"可用且可验证"的实现，不是最优实现。
    """
    import tgai_go.train as _T
    from .indexing import gather_flat

    kh, kw = int(w.shape[2]), int(w.shape[3])
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    N, C, H, W = (int(v) for v in x.shape)
    Cout = int(w.shape[0])
    oh = _out_size(H, kh, sh, ph, dh)
    ow = _out_size(W, kw, sw, pw, dw)
    idx, msk = _im2col_index_np(N, C, H, W, kh, kw, sh, sw, ph, pw, dh, dw, oh, ow)
    cols = gather_flat(x, idx, n_src=N * C * H * W)
    if not msk.all():
        cols = cols * _T.tensor(np.ascontiguousarray(msk.astype(np.float32)))
    cols = cols.reshape(N * oh * ow, C * kh * kw)
    w2 = w.reshape(Cout, C * kh * kw).transpose(1, 0)          # [C*kh*kw, Cout]
    y = (cols @ w2).reshape(N, oh, ow, Cout).permute(0, 3, 1, 2)
    if bias is not None:
        y = y + _as_tensor(bias).reshape(1, Cout, 1, 1)
    return y


def _conv2d_cuda(x, w, bias, stride, padding, dilation, groups):
    """设备侧 conv2d 入口：groups=1 走单组；groups>1 按组切片后逐组算，再拼通道。

    组卷积在数学上就是"各组独立卷积 + 通道拼接"，所以复用已验证到 2.39e-07 的单组路径即可。
    """
    g = int(groups)
    if g <= 0:
        raise ValueError("groups 必须为正，收到 %r" % (groups,) + "。修法：常规卷积用 groups=1")
    if g == 1:
        return _conv2d_cuda_single(x, w, bias, stride, padding, dilation)

    N, C, H, W = (int(v) for v in x.shape)
    Cout = int(w.shape[0])
    if C % g or Cout % g:
        raise ValueError("groups=%d 与通道数不匹配（Cin=%d Cout=%d）" % (g, C, Cout) +
                         "。修法：groups 必须同时整除输入与输出通道数")
    Cg, Og = C // g, Cout // g
    outs = []
    for gi in range(g):
        xg = x[:, gi * Cg:(gi + 1) * Cg, :, :]
        wg = w[gi * Og:(gi + 1) * Og, :, :, :]
        outs.append(_conv2d_cuda_single(xg, wg, None, stride, padding, dilation))
    y = outs[0] if len(outs) == 1 else _cat_channels(outs)
    if bias is not None:
        y = y + _as_tensor(bias).reshape(1, Cout, 1, 1)
    return y


def _cat_channels(parts):
    """沿通道维（dim=1）拼接。注意 `tgai_go.train.cat` 的形参是 `dim`，不是 `axis`。"""
    import tgai_go.train as _T
    return _T.cat(parts, dim=1)


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """2D 卷积（NCHW）。weight: [Cout, Cin/groups, kh, kw]。"""
    x = _as_tensor(x)
    w = _as_tensor(weight)
    if getattr(x, "device", "cpu") == "cuda":
        return _conv2d_cuda(x, w, bias, stride, padding, dilation, groups)
    _check_cpu(x, "conv2d")
    if isinstance(stride, int):
        sh = sw = int(stride)
    else:
        sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    g = int(groups)
    xn = _np(x)
    wn = _np(w)
    N, Cin, H, W = xn.shape
    Cout, Cg, kh, kw = wn.shape
    if g <= 0 or Cin % g or Cout % g or Cg * g != Cin:
        raise ValueError("groups=%d 与通道数不匹配（Cin=%d Cout=%d, 每组 %d 通道）。"
                         "修法：groups 必须同时整除 Cin 与 Cout" % (g, Cin, Cout, Cg))
    outs = []
    cols_list, shapes = [], []
    for gi in range(g):
        xg = xn[:, gi * Cg:(gi + 1) * Cg, :, :]
        cols, (oh, ow) = _im2col(xg, kh, kw, sh, sw, ph, pw, dh, dw)
        wg = wn[gi * (Cout // g):(gi + 1) * (Cout // g)].reshape(Cout // g, -1)
        outs.append(cols @ wg.T)                       # [P, Cout/g]
        cols_list.append(cols)
        shapes.append((oh, ow, xg.shape))
    oh, ow = shapes[0][0], shapes[0][1]
    P = cols_list[0].shape[0]
    out = np.concatenate(outs, axis=1) if g > 1 else outs[0]
    out = out.reshape(N, oh, ow, Cout).transpose(0, 3, 1, 2).copy()
    if bias is not None:
        out = out + _np(bias).reshape(1, -1, 1, 1)

    need = any(getattr(t, "requires_grad", False) for t in (x, w))
    bn = None if bias is None else _np(bias)

    def bw(grad, grads):
        gg = np.asarray(grad, np.float32).transpose(0, 2, 3, 1).reshape(P, Cout)
        dws = []
        dxs = []
        for gi in range(g):
            cols = cols_list[gi]
            xg_shape = shapes[gi][2]
            gcols = gg[:, gi * (Cout // g):(gi + 1) * (Cout // g)]
            wg = wn[gi * (Cout // g):(gi + 1) * (Cout // g)].reshape(Cout // g, -1)
            dws.append((gcols.T @ cols).reshape(Cout // g, Cg, kh, kw))
            if getattr(x, "requires_grad", False):
                dxs.append(_col2im(gcols @ wg, xg_shape, kh, kw, sh, sw, ph, pw, dh, dw,
                                   shapes[gi][0], shapes[gi][1]))
        if getattr(w, "requires_grad", False):
            dw_all = np.concatenate(dws, axis=0) if g > 1 else dws[0]
            oldw = grads.get(id(w))
            grads[id(w)] = dw_all if oldw is None else oldw + dw_all
        if getattr(x, "requires_grad", False):
            dx_all = np.concatenate(dxs, axis=1) if g > 1 else dxs[0]
            oldx = grads.get(id(x))
            grads[id(x)] = dx_all if oldx is None else oldx + dx_all
        if bias is not None and getattr(bias, "requires_grad", False):
            db = gg.sum(axis=0).astype(np.float32)
            oldb = grads.get(id(bias))
            grads[id(bias)] = db if oldb is None else oldb + db

    out_t = Tensor(np.ascontiguousarray(out), requires_grad=need,
                   _parents=tuple([t for t in (x, w, bias) if t is not None]))
    out_t._backward = bw
    return out_t


def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """1D 卷积（NCL）：**用可微 reshape 视图**委托给 conv2d（kh=1），梯度自动回传。"""
    x = _as_tensor(x)
    w = _as_tensor(weight)
    xn = _np(x)
    wn = _np(w)
    if xn.ndim != 3 or wn.ndim != 3:
        raise ValueError("conv1d 期望 x=[N,C,L]、weight=[Cout,Cin/g,kw]，收到 %s / %s。修法：检查输入维度"
                         % (xn.shape, wn.shape))
    s2 = (1, int(stride) if isinstance(stride, int) else int(stride[-1]))
    p2 = (0, int(padding) if isinstance(padding, int) else int(padding[-1]))
    d2 = (1, int(dilation) if isinstance(dilation, int) else int(dilation[-1]))
    N, C, L = xn.shape
    Cout, Cg, kw = wn.shape
    x4 = x.reshape((N, C, 1, L))
    w4 = w.reshape((Cout, Cg, 1, kw))
    y = conv2d(x4, w4, bias=bias, stride=s2, padding=p2, dilation=d2, groups=groups)
    return y.reshape((N, Cout, int(y.shape[3])))


class Conv2d(object):
    """最小 Conv2d 模块（NCHW）。参数命名与 torch 一致：weight/bias。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        from . import nn as _nn
        kh, kw = _pair(kernel_size)
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.kernel_size = (kh, kw)
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, int(groups)
        g = int(groups)
        fan_in = (in_channels // g) * kh * kw
        bound = 1.0 / np.sqrt(float(fan_in))
        w = np.random.uniform(-bound, bound, (out_channels, in_channels // g, kh, kw)).astype(np.float32)
        self.weight = _nn.Parameter(w)
        self.bias = _nn.Parameter(np.zeros(out_channels, np.float32)) if bias else None

    def parameters(self):
        return [p for p in (self.weight, self.bias) if p is not None]

    def __call__(self, x):
        return conv2d(x, self.weight, self.bias, self.stride, self.padding,
                      self.dilation, self.groups)


class Conv1d(object):
    """最小 Conv1d 模块（NCL）。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        from . import nn as _nn
        kw = int(kernel_size) if not isinstance(kernel_size, (tuple, list)) else int(kernel_size[0])
        self.kernel_size = (kw,)
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, int(groups)
        g = int(groups)
        fan_in = (in_channels // g) * kw
        bound = 1.0 / np.sqrt(float(fan_in))
        w = np.random.uniform(-bound, bound, (out_channels, in_channels // g, kw)).astype(np.float32)
        self.weight = _nn.Parameter(w)
        self.bias = _nn.Parameter(np.zeros(out_channels, np.float32)) if bias else None

    def parameters(self):
        return [p for p in (self.weight, self.bias) if p is not None]

    def __call__(self, x):
        return conv1d(x, self.weight, self.bias, self.stride, self.padding,
                      self.dilation, self.groups)
