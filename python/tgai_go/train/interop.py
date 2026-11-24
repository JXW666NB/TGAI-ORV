# -*- coding: utf-8 -*-
"""与 `torch.nn` 的**权重级**互操作：`from_torch` / `to_torch` / state_dict 互转。

边界（先说清楚，避免误会）
* 这是**权重级**互转：把 torch 层的参数搬进等价的 ORV 层（或反过来），**不是**运行期互操作 ——
  不能在 ORV 的计算图里直接调用 torch 的层，也不共享 autograd。
* 支持的层（0.9.21 起）：`nn.Linear`、`nn.Embedding`、`nn.LayerNorm`、`nn.RMSNorm`、
  `nn.Conv1d`、`nn.Conv2d`、`nn.Dropout`、`nn.BatchNorm1d/2d`、`nn.MaxPool1d/2d`、
  `nn.AvgPool1d/2d`；容器支持 `nn.Sequential` 与 `nn.ModuleList`。
* **不支持**自定义 `Module` 子类：它的 `forward` 是任意 Python，无法自动翻译。
  迁移时把它的 `forward` 用 ORV 的层重写一遍即可（两者数学逐算子对齐）。
* 说明：本文档一度写着"LayerNorm 不支持，因为 ORV 没有 LayerNorm 实现"，那是**过时信息** ——
  LayerNorm 早已实现，`from_torch` 也一直在转它；0.9.21 修正了这段文案与报错信息。
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
    if name == "Conv1d":
        out = T.Conv1d(int(module.in_channels), int(module.out_channels),
                          int(module.kernel_size[0]),
                          stride=int(module.stride[0]), padding=int(module.padding[0]),
                          dilation=int(module.dilation[0]), groups=int(module.groups),
                          bias=module.bias is not None)
        _set_param(out.weight, _np(module.weight))
        if module.bias is not None:
            _set_param(out.bias, _np(module.bias))
        return out
    if name == "Dropout":
        # 训练/推理行为由 Module.train() 控制，概率直接搬
        return T.nn.Dropout(float(module.p))
    if name in ("BatchNorm1d", "BatchNorm2d"):
        dim = 1 if name == "BatchNorm1d" else 2
        out = (T.nn.BatchNorm1d if dim == 1 else T.nn.BatchNorm2d)(
            int(module.num_features), eps=float(module.eps), momentum=float(module.momentum),
            affine=bool(module.affine))
        if module.affine:
            _set_param(out.weight, _np(module.weight))
            _set_param(out.bias, _np(module.bias))
        out.running_mean = _np(module.running_mean)
        out.running_var = _np(module.running_var)
        return out
    if name in ("MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d"):
        k = module.kernel_size
        kk = int(k) if isinstance(k, int) else int(k[0])
        st = module.stride
        ss = None if st is None else (int(st) if isinstance(st, int) else int(st[0]))
        pd = module.padding
        pp = int(pd) if isinstance(pd, int) else int(pd[0])
        cls = {"MaxPool1d": T.nn.MaxPool1d, "MaxPool2d": T.nn.MaxPool2d,
               "AvgPool1d": T.nn.AvgPool1d, "AvgPool2d": T.nn.AvgPool2d}[name]
        return cls(kk, stride=ss, padding=pp)
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
        "from_torch 不支持 %s。已支持：Linear / Embedding / LayerNorm / RMSNorm / "
        "Conv1d / Conv2d / Dropout / BatchNorm1d / BatchNorm2d / MaxPool1d / MaxPool2d / "
        "AvgPool1d / AvgPool2d / Sequential / ModuleList。"
        "修法：自定义 Module 的 forward 无法自动复现，请把它的 forward 用 ORV 的层重写一遍"
        "（数学与 PyT 逐算子对齐，见 docs/REVIEW 的对照表）" % name)


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
