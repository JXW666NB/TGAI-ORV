# -*- coding: utf-8 -*-
"""与 `torch.nn` 的**权重级**互操作：`from_torch` / `to_torch` / state_dict 互转。

边界（先说清楚，避免误会）
* 这是**权重级**互转：把 torch 层的参数搬进等价的 ORV 层（或反过来），**不是**运行期互操作 ——
  不能在 ORV 的计算图里直接调用 torch 的层，也不共享 autograd。
* 支持的层：`nn.Linear`、`nn.Embedding`、`nn.RMSNorm`、`Conv2d`（本库的 `T.Conv2d`）。
  **`nn.LayerNorm` 暂不支持**，因为 ORV 目前没有 LayerNorm 实现（这是与 PyT 对齐的真实缺口）。
* 容器只支持 `nn.Sequential`（顺序执行，语义可直接重放）。自定义 `nn.Module` 子类的 forward
  无法自动复现，遇到会明确报错而不是猜。
"""
from __future__ import annotations

import numpy as np

__all__ = ["from_torch", "to_torch", "state_dict_from_torch", "state_dict_to_torch",
           "OrvSequential", "InteropError"]


class InteropError(TypeError):
    """不支持的类型/结构。"""


def _np(t):
    return np.asarray(t.detach().cpu().numpy(), np.float32)


def _set_param(param, arr):
    dst = np.asarray(param.data, np.float32)
    if tuple(dst.shape) != tuple(arr.shape):
        raise ValueError("参数形状不匹配：ORV %s vs 源 %s。修法：确认两边是同一层的同一参数"
                         % (dst.shape, arr.shape))
    np.copyto(dst, np.ascontiguousarray(arr, np.float32))


def _torch_type_name(m):
    return type(m).__name__


class OrvSequential(object):
    """顺序容器（对应 nn.Sequential）：forward 依次调用。"""

    def __init__(self, layers):
        self.layers = list(layers)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def parameters(self):
        out = []
        for m in self.layers:
            out.extend(m.parameters())
        return out

    def __call__(self, x):
        for m in self.layers:
            x = m(x)
        return x


def from_torch(module):
    """torch 层/Sequential → ORV 等价层。"""
    import tgai_go.train as T

    name = _torch_type_name(module)
    if name == "Sequential" or name == "ModuleList":
        return OrvSequential([from_torch(m) for m in module])
    if name == "Linear":
        out = T.nn.Linear(int(module.in_features), int(module.out_features),
                          bias=module.bias is not None)
        _set_param(out.weight, _np(module.weight))
        if module.bias is not None:
            _set_param(out.bias, _np(module.bias))
        return out
    if name == "Embedding":
        out = T.nn.Embedding(int(module.num_embeddings), int(module.embedding_dim))
        _set_param(out.weight, _np(module.weight))
        return out
    if name == "LayerNorm":
        ns = module.normalized_shape
        dim = int(ns) if isinstance(ns, int) else int(ns[-1])
        out = T.nn.LayerNorm(dim, eps=float(module.eps),
                             elementwise_affine=bool(module.elementwise_affine))
        if getattr(module, "weight", None) is not None:
            _set_param(out.weight, _np(module.weight))
            _set_param(out.bias, _np(module.bias))
        return out
    if name == "RMSNorm":
        out = T.nn.RMSNorm(int(module.normalized_shape) if isinstance(module.normalized_shape, int)
                           else int(module.normalized_shape[0]),
                           eps=float(module.eps) if module.eps is not None else 1e-6)
        if getattr(module, "weight", None) is not None:
            _set_param(out.weight, _np(module.weight))
        return out
    if name == "Conv2d":
        out = T.Conv2d(int(module.in_channels), int(module.out_channels),
                       (int(module.kernel_size[0]), int(module.kernel_size[1])),
                       stride=(int(module.stride[0]), int(module.stride[1])),
                       padding=(int(module.padding[0]), int(module.padding[1])),
                       dilation=(int(module.dilation[0]), int(module.dilation[1])),
                       groups=int(module.groups), bias=module.bias is not None)
        _set_param(out.weight, _np(module.weight))
        if module.bias is not None:
            _set_param(out.bias, _np(module.bias))
        return out
    raise InteropError(
        "from_torch 不支持 %s。修法：只用 Linear/Embedding/RMSNorm/Conv2d 或 nn.Sequential；"
        "LayerNorm 暂不支持（ORV 还没有 LayerNorm 实现），自定义 Module 的 forward 无法自动复现"
        % name)


def to_torch(module):
    """ORV 层/OrvSequential → torch 等价层（参数逐位复制）。"""
    import torch
    import torch.nn as tn

    import tgai_go.train as T

    if isinstance(module, OrvSequential):
        return tn.Sequential(*[to_torch(m) for m in module.layers])
    if isinstance(module, T.Conv2d):
        m = tn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                      stride=module.stride, padding=module.padding, dilation=module.dilation,
                      groups=module.groups, bias=module.bias is not None)
        with torch.no_grad():
            m.weight.copy_(torch.from_numpy(np.asarray(module.weight.data, np.float32)))
            if module.bias is not None:
                m.bias.copy_(torch.from_numpy(np.asarray(module.bias.data, np.float32)))
        return m
    cls = _torch_type_name(module)
    if cls == "Linear":
        m = tn.Linear(int(module.weight.shape[1]), int(module.weight.shape[0]),
                      bias=getattr(module, "bias", None) is not None)
        with torch.no_grad():
            m.weight.copy_(torch.from_numpy(np.asarray(module.weight.data, np.float32)))
            if module.bias is not None:
                m.bias.copy_(torch.from_numpy(np.asarray(module.bias.data, np.float32)))
        return m
    if cls == "Embedding":
        m = tn.Embedding(int(module.weight.shape[0]), int(module.weight.shape[1]))
        with torch.no_grad():
            m.weight.copy_(torch.from_numpy(np.asarray(module.weight.data, np.float32)))
        return m
    if cls == "LayerNorm":
        dim = int(module.normalized_shape[0])
        m = tn.LayerNorm(dim, eps=float(module.eps),
                         elementwise_affine=bool(module.elementwise_affine))
        with torch.no_grad():
            if module.weight is not None:
                m.weight.copy_(torch.from_numpy(np.asarray(module.weight.data, np.float32)))
                m.bias.copy_(torch.from_numpy(np.asarray(module.bias.data, np.float32)))
        return m
    if cls == "RMSNorm":
        dim = int(module.weight.shape[-1]) if getattr(module, "weight", None) is not None else 1
        m = tn.RMSNorm(dim, eps=float(getattr(module, "eps", 1e-6)))
        with torch.no_grad():
            if getattr(module, "weight", None) is not None:
                m.weight.copy_(torch.from_numpy(np.asarray(module.weight.data, np.float32)))
        return m
    raise InteropError("to_torch 不支持 %s。修法：只用 Linear/Embedding/RMSNorm/Conv2d 或 "
                       "OrvSequential（from_torch 的产物）" % cls)


def state_dict_from_torch(module):
    """torch 的 state_dict → {名字: numpy}，名字**原样保留**（便于对照 torch 的键）。"""
    out = {}
    for k, v in module.state_dict().items():
        out[k] = _np(v)
    return out


def state_dict_to_torch(orv_module_or_seq, prefix=""):
    """ORV 层/Sequential → {名字: torch.Tensor}，命名与 torch 的 Sequential 一致（0.weight…）。"""
    import torch

    out = {}
    if isinstance(orv_module_or_seq, OrvSequential):
        for i, m in enumerate(orv_module_or_seq.layers):
            out.update(state_dict_to_torch(m, "%s%d." % (prefix, i)))
        return out
    m = orv_module_or_seq
    if getattr(m, "weight", None) is not None:
        out[prefix + "weight"] = torch.from_numpy(
            np.asarray(m.weight.data, np.float32).copy())
    if getattr(m, "bias", None) is not None:
        out[prefix + "bias"] = torch.from_numpy(
            np.asarray(m.bias.data, np.float32).copy())
    return out
