# -*- coding: utf-8 -*-
"""可微的索引算子：`gather` / `index_select`（对齐 torch 语义）。

为什么要它：
  * 阶段 B 要求补 gather；它同时解决两处**每步 134 MB 的主机往返**（embedding 行取值、
    embedding 梯度累加）和 `align.token_logprobs` 的 `[T,V]` one-hot。
  * 后端原先只有**仿射索引计划**（base/div/stride，用于 slice/permute/cat），
    表达不了任意索引，所以新增了 `gather_rows` / `gather_flat` 内核与对应的 atomicAdd 反向。

语义（与 torch 一致的部分）：
  * `index_select(x, dim, index)`：沿 dim 取，index 是 1 维整数数组；
  * `gather_flat(x, flat_index)`：`out[i] = x.reshape(-1)[flat_index[i]]`（对应 torch.gather
    的扁平化用法）。

数值：CPU 路径走 numpy；CUDA 路径走新内核（反向 atomicAdd）。
"""
import numpy as np

from . import Tensor, _as_tensor, _cuda

__all__ = ["index_select", "gather_flat", "scatter_flat"]


def _is_cuda(t):
    return getattr(t, "device", "cpu") == "cuda"


def _np_of(t):
    return np.asarray(t.to_numpy() if hasattr(t, "to_numpy") else t.data, np.float32)


def _out_tensor(arr, parents, requires_grad, backward):
    out = Tensor(np.ascontiguousarray(arr, np.float32), requires_grad=requires_grad,
                 _parents=tuple(parents))
    out._backward = backward
    return out


def index_select(x, dim, index):
    """沿 dim 取行（torch.index_select 语义）。反向把梯度 atomicAdd 回原位置。"""
    x = _as_tensor(x)
    idx = np.asarray(index, np.int64).reshape(-1)
    shape = tuple(int(s) for s in x.shape)
    d = int(dim)
    if d < 0:
        d += len(shape)
    if not (0 <= d < len(shape)):
        raise ValueError("index_select 的 dim=%d 越界（shape=%s）。修法：dim 必须在 [0, %d)"
                         % (dim, shape, len(shape)))
    n_src = shape[d]
    if idx.size and (idx.min() < 0 or idx.max() >= n_src):
        raise ValueError("index_select 下标越界：范围 [%d, %d]，该维长度 %d。修法：检查索引来源"
                         % (int(idx.min()), int(idx.max()), n_src))
    outer = int(np.prod(shape[:d])) if d else 1
    inner = int(np.prod(shape[d + 1:])) if d + 1 < len(shape) else 1
    out_rows = idx.size
    out_shape = shape[:d] + (out_rows,) + shape[d + 1:]
    need = bool(getattr(x, "requires_grad", False))

    if not _is_cuda(x):
        arr = _np_of(x).reshape(outer, n_src, inner)
        gathered = arr[:, idx, :].reshape(out_shape)

        def bw(g, grads):
            gg = np.asarray(g, np.float32).reshape(outer, out_rows, inner)
            dx = np.zeros((outer, n_src, inner), np.float32)
            np.add.at(dx, (slice(None), idx, slice(None)), gg)
            ddx = dx.reshape(shape)
            old = grads.get(id(x))
            grads[id(x)] = ddx if old is None else old + ddx

        return _out_tensor(gathered, [x], need, bw)

    rt = _cuda()
    id_buf = rt.h2d(idx.astype(np.int32), dtype=np.int32)
    src = x._gpu
    # 沿 dim 取值：先按 (outer, n_src, inner) 的行形态做 outer*inner 次行 gather 不划算，
    # 这里用「等价重排」——把 dim 移到最外再用行 gather：此处只支持 outer==1 的常见情形
    # （Embedding / 词表行取值 / 批次首维），其余情况先转置到主机再算并注明。
    if outer == 1:
        out_buf = rt.gather_rows(src, id_buf, n_src, inner, out_rows)

        def bw_c(g, grads):
            # CUDA 路径下 g 直接是 GpuBuf（不是 Tensor），别包成 Tensor
            dxbuf = rt.scatter_add_rows(g, id_buf, n_src, inner, out_rows)
            old = grads.get(id(x))
            if old is None:
                grads[id(x)] = dxbuf
            else:
                grads[id(x)] = rt.add(old, dxbuf)

        out = Tensor._from_gpu(out_buf, requires_grad=need, _parents=(x,))
        out._backward = bw_c
        return out

    # 兜底：outer>1 时走主机（正确但慢；文档已注明）
    arr = _np_of(x).reshape(outer, n_src, inner)
    gathered = arr[:, idx, :].reshape(out_shape)

    def bw_h(g, grads):
        gg = np.asarray(g, np.float32).reshape(outer, out_rows, inner)
        dx = np.zeros((outer, n_src, inner), np.float32)
        np.add.at(dx, (slice(None), idx, slice(None)), gg)
        ddx = dx.reshape(shape)
        old = grads.get(id(x))
        grads[id(x)] = ddx if old is None else old + ddx

    return _out_tensor(gathered, [x], need, bw_h)


def gather_flat(x, flat_index, n_src=None):
    """逐元素 gather：out[i] = x.reshape(-1)[flat_index[i]]（反向 atomicAdd 回原扁平位置）。"""
    x = _as_tensor(x)
    idx = np.asarray(flat_index, np.int64).reshape(-1)
    n = int(np.prod([int(s) for s in x.shape])) if len(x.shape) else 1
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ValueError("gather_flat 下标越界：范围 [%d, %d]，扁平长度 %d。修法：检查索引来源"
                         % (int(idx.min()), int(idx.max()), n))
    need = bool(getattr(x, "requires_grad", False))

    if not _is_cuda(x):
        flat = _np_of(x).reshape(-1)
        out = flat[idx]

        xshape = tuple(int(v) for v in x.shape)

        def bw(g, grads):
            gg = np.asarray(g, np.float32)
            dx = np.zeros(n, np.float32)
            np.add.at(dx, idx, gg)
            dx = dx.reshape(xshape)          # 必须还原成参数形状
            old = grads.get(id(x))
            grads[id(x)] = dx if old is None else old + dx

        return _out_tensor(out, [x], need, bw)

    rt = _cuda()
    id_buf = rt.h2d(idx.astype(np.int32), dtype=np.int32)
    out_buf = rt.gather_flat(x._gpu, id_buf, idx.size)

    xshape = tuple(int(v) for v in x.shape)

    def bw_c(g, grads):
        # CUDA 路径下 g 直接是 GpuBuf
        buf = rt.scatter_add_flat(g, id_buf, n, idx.size)
        dxbuf = Tensor._from_gpu(buf, requires_grad=False).reshape(xshape)._gpu
        old = grads.get(id(x))
        if old is None:
            grads[id(x)] = dxbuf
        else:
            grads[id(x)] = rt.add(old, dxbuf)

    out = Tensor._from_gpu(out_buf, requires_grad=need, _parents=(x,))
    out._backward = bw_c
    return out


def scatter_flat(src, flat_index, n_out):
    """`gather_flat` 的转置：out = zeros(n_out); out[idx] += src（重复下标累加）。

    反向就是 `gather_flat(g, idx)` —— 这是把「逐 token 取到的值放回 [T] 位置」的关键，
    它让 `token_logprobs` 不必再造一份 [T, V] 的 one-hot。
    """
    x = _as_tensor(src)
    idx = np.asarray(flat_index, np.int64).reshape(-1)
    n_out = int(n_out)
    n = int(np.prod([int(s) for s in x.shape])) if len(x.shape) else 1
    if idx.size != n:
        raise ValueError("scatter_flat 的索引数 %d 与源元素数 %d 不一致。修法：两者必须一一对应"
                         % (idx.size, n))
    if idx.size and (idx.min() < 0 or idx.max() >= n_out):
        raise ValueError("scatter_flat 下标越界：范围 [%d, %d]，输出长度 %d。修法：检查索引来源"
                         % (int(idx.min()), int(idx.max()), n_out))
    need = bool(getattr(x, "requires_grad", False))
    out_shape = (n_out,)

    if not _is_cuda(x):
        def bw(g, grads):
            gg = np.asarray(g, np.float32).reshape(-1)[idx]
            old = grads.get(id(x))
            grads[id(x)] = gg if old is None else old + gg

        arr = np.zeros(n_out, np.float32)
        np.add.at(arr, idx, _np_of(x).reshape(-1))
        return _out_tensor(arr, [x], need, bw)

    rt = _cuda()
    id_buf = rt.h2d(idx.astype(np.int32), dtype=np.int32)
    buf = rt.scatter_add_flat(x._gpu, id_buf, n_out, idx.size)

    def bw_c(g, grads):
        picked = rt.gather_flat(g, id_buf, idx.size)
        old = grads.get(id(x))
        if old is None:
            grads[id(x)] = picked
        else:
            grads[id(x)] = rt.add(old, picked)

    out = Tensor._from_gpu(buf, requires_grad=need, _parents=(x,))
    out._backward = bw_c
    return out
