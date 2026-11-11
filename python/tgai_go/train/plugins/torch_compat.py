# -*- coding: utf-8 -*-
"""示例插件：把 `torch` 这个名字指向 ORV。

## 这个插件证明什么

「**PyTorch 的 API 面本身可以是插件**」—— 核心不背这个包袱，社区一个文件就能补上。

装法（三种任选）：

```python
import tgai_go.train as T
T.plugin.load("examples/plugins/torch_compat.py")   # 从文件装
T.plugin.activate("torch-compat")                   # 显式激活（会改 sys.modules）
```

之后老代码**一行不改**：

```python
import torch
import torch.nn as nn
m = nn.Linear(4, 8)
```

## 安全边界（重要）

* **绝不覆盖真的 torch**：若 `sys.modules["torch"]` 已被真正的 torch 占用，本插件**拒绝安装**，
  而不是悄悄替换 —— 那会让用户以为在用 PyTorch 而实际跑的是 ORV。
* 卸载用 `plugin` 的 `uninstall_torch_compat()`（见文件末尾），把 `sys.modules` 复原。

## 覆盖范围（诚实声明）

本示例只覆盖**最小可运行闭环**，不是完整兼容层：

| 已覆盖 | 未覆盖 |
| --- | --- |
| `torch.tensor / zeros / ones / randn / from_numpy` | `torchvision` / `torchaudio` 等生态包 |
| `torch.Tensor`、`torch.float32` 等 dtype | `torch.compile`、`torch.jit` |
| `torch.nn`（Module / Linear / Embedding / LayerNorm / RMSNorm / Sequential / ModuleList / functional） | 二阶导数、`torch.func` |
| `torch.optim`（AdamW / SGD / Adafactor / LAMB / Lion / Muon） | 分布式（DDP / FSDP） |
| `torch.no_grad`、`torch.manual_seed` | CUDA Graph 的公开 API |

要更完整，社区可以在这个文件上继续加 —— 这正是插件系统的用途。
"""

import sys
import types

PLUGIN = {
    "name": "torch-compat",
    "version": "0.1.0",
    "api_version": "1.0",
    "ops": [],           # 本插件不注册新算子，只装 API 别名
}

#: 标记，用来识别"这个 torch 是我们装的"
_MARK = "__tgai_torch_compat__"


def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    setattr(m, _MARK, True)
    return m


def _build():
    import numpy as np

    import tgai_go.train as T

    # ---- torch.nn.functional ----
    fn = T.nn.functional
    nn_functional = _mod("torch.nn.functional", **{
        k: getattr(fn, k) for k in dir(fn) if not k.startswith("_")
    })

    # ---- torch.nn ----
    nn = _mod("torch.nn", functional=nn_functional)
    for k in ("Module", "Linear", "Embedding", "LayerNorm", "RMSNorm",
              "Sequential", "ModuleList", "Parameter", "Conv1d", "Conv2d",
              "Dropout", "BatchNorm1d", "BatchNorm2d", "MoE"):
        if hasattr(T.nn, k):
            setattr(nn, k, getattr(T.nn, k))

    # ---- torch.optim ----
    optim = _mod("torch.optim")
    for k in ("AdamW", "SGD", "Adafactor", "LAMB", "Lion", "Muon"):
        if hasattr(T.optim, k):
            setattr(optim, k, getattr(T.optim, k))

    # ---- torch.cuda ----
    def _cuda_available():
        try:
            T._cuda()
            return True
        except Exception:      # noqa: BLE001
            return False

    cuda = _mod("torch.cuda", is_available=_cuda_available, device_count=lambda: 1)

    # ---- 顶层 torch ----
    top = _mod(
        "torch",
        tensor=T.tensor,
        zeros=T.zeros, ones=T.ones,
        from_numpy=T.tensor,
        randn=lambda *shape, **kw: T.tensor(
            np.random.default_rng(kw.get("seed")).normal(0, 1, shape).astype(np.float32)),
        Tensor=T.Tensor,
        nn=nn, optim=optim, cuda=cuda,
        no_grad=T.no_grad,
        manual_seed=lambda seed: np.random.seed(int(seed)),
        float32=np.float32, float16=np.float16, int64=np.int64,
        __version__="%s+tgai-compat" % T.__dict__.get("__version__", "0"),
    )
    return top, nn, optim, nn_functional, cuda


def install():
    """把 `torch` 指向 ORV。**真 torch 已加载则拒绝**。"""
    existing = sys.modules.get("torch")
    if existing is not None and not getattr(existing, _MARK, False):
        raise RuntimeError(
            "已加载真正的 torch（%r），拒绝覆盖" % getattr(existing, "__file__", "?") +
            "。修法：本插件只在没装/没导入 torch 的环境里用；"
            "要在同一进程里对比两者，请起两个进程分别测")
    top, nn, optim, nn_functional, cuda = _build()
    sys.modules["torch"] = top
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = nn_functional
    sys.modules["torch.optim"] = optim
    sys.modules["torch.cuda"] = cuda
    return top


def uninstall_torch_compat():
    """把 sys.modules 复原（幂等）。"""
    for k in ("torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.cuda"):
        m = sys.modules.get(k)
        if m is not None and getattr(m, _MARK, False):
            del sys.modules[k]
    return None


def _register():
    from tgai_go.train import plugin
    # replace=True：同一插件被重复 load / 重新 import 时要幂等，不能报重名
    plugin.api("torch-compat", install, replace=True)


# 被 `plugin.load(...)` 加载时会执行到这里：只登记、不安装（安装需显式 activate）
_register()
