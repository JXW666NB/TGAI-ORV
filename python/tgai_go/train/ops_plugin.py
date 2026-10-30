# -*- coding: utf-8 -*-
"""算子插件系统 —— 「核心精简，一切皆插件」。

## 为什么要有它

我们是 **LLM 训练引擎**，核心只覆盖训练必需的那几十个算子（见 `docs/OP_COVERAGE.md`）。
但别人可能要：
* 图形学：`conv2d`、`grid_sample`、法线/光照类算子；
* 科研：`fft`、`svd`、特殊函数（`erf`、`bessel_j0`）；
* 自家模型的私有算子：门控、稀疏路由、自定义损失。

这些**不该塞进核心**（会让内核越滚越大、每个后端都要跟着实现），
而应该由需要的人**以插件形式加进来** —— 用完即走，核心不受影响。

## 两种插件（按难度从低到高）

### A. 组合式（推荐，几乎不用写数学）
只由已有 Tensor 算子拼出来 ⇒ **autograd 自动生效**，不用写反向。

```python
from tgai_go.train import ops_plugin, Tensor

def softplus(x):
    return (x.exp() + 1.0).log()          # 全用已有算子

ops_plugin.register("softplus", softplus)
# 之后：T.functional.softplus(x) 或  x.softplus()
```

### B. 自带反向（要自己写数学，性能优先）
给出 numpy 前向 + 反向（返回每个输入父节点的梯度）：

```python
import numpy as np

def gelu_exact(x):                       # numpy 前向
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def gelu_exact_bwd(g, out, xs):          # 反向：返回 [dL/dx]
    x, = xs
    pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    return [g * (cdf + x * pdf)]

ops_plugin.register_numpy("gelu_exact", gelu_exact, gelu_exact_bwd)
```

## 怎么被发现（三种，任选）

1. **手工注册**：`import` 你的模块即可；
2. **环境变量**：把 `.py` 放到某目录，设 `TG_OPS_PATH=/path/to/dir[:/path2]`，导入引擎时自动加载；
3. **包入口点**（推荐发布给别人用时）：
   ```toml
   [project.entry-points."tgai_go.ops"]
   my_ops = "my_pkg.ops:register_all"
   ```
   `pip install` 后自动被发现。

## 规则（写在这里，免得踩坑）

* **不许覆盖核心算子**：除非显式 `replace=True`，重名直接报错；
  「覆盖核心」是被允许的（有人想拿更快的实现替换），但必须自己声明，且会被记录。
* 插件是**用户代码**，引擎不会审计它的正确性 —— 数值对不对由插件的测试负责。
  自检命令可以列出当前加载了哪些插件（`ops_plugin.list_ops()`）。
* 插件**不改变**后端契约：组合式插件自动适配所有后端；自带反向的插件用的是 numpy，
  对 CUDA/CPU 都能跑，但绕过了融合核（性能另说）。
* 卸载：`unregister(name)`，用于实验对比或回滚。
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

from . import Tensor, _acc
from . import functional as F

#: name -> dict(kind, fn, backward, doc, source, replaced_core)
_REGISTRY = {}
#: 记录被插件替换掉的核心名字（透明化）
_REPLACED_CORE = set()

_CORE_NAMES = None


def _core_names():
    """核心已提供的名字（用于判断是否在覆盖核心）。"""
    global _CORE_NAMES
    if _CORE_NAMES is None:
        names = {n for n in dir(F) if not n.startswith("_")}
        names |= {n for n in dir(Tensor) if not n.startswith("_")}
        _CORE_NAMES = names
    return _CORE_NAMES


def register(name, fn, *, kind="functional", doc="", replace=False):
    """注册**组合式**算子（fn 只用已有 Tensor 算子实现，autograd 自动生效）。

    kind="functional" ⇒ 挂到 `tgai_go.train.functional.<name>`
    kind="tensor"     ⇒ 同时挂成 `Tensor.<name>` 方法（第一个参数是 self）
    """
    name = str(name).strip()
    if not name.isidentifier():
        raise ValueError("算子名必须是合法标识符：%r。修法：用字母/下划线开头，如 'my_op'" % (name,))
    if not callable(fn):
        raise TypeError("算子 %r 的实现必须可调用（Tensor -> Tensor）" % (name,))
    if name in _REGISTRY and not replace:
        raise ValueError(
            "算子 %r 已注册（来源：%s）。修法：换一个名字，或显式 replace=True 覆盖"
            % (name, _REGISTRY[name].get("source", "?")))
    if name in _core_names() and not replace:
        raise ValueError(
            "算子 %r 与核心算子重名。修法：换名字；确实要替换核心实现请显式 replace=True"
            % (name,))
    if name in _core_names():
        _REPLACED_CORE.add(name)

    _REGISTRY[name] = dict(kind=kind, fn=fn, backward=None, doc=doc, source="manual",
                           replaced_core=name in _core_names())
    setattr(F, name, fn)
    if kind == "tensor":
        setattr(Tensor, name, fn)
    return fn


def register_numpy(name, forward, backward, *, kind="functional", doc="", replace=False):
    """注册**自带反向**的算子。

    forward(*np_arrays, **kwargs) -> np.ndarray
    backward(g, out, xs) -> list[np.ndarray]（与 xs 一一对应；不需要梯度的位置给 None）
    """
    if not callable(forward) or not callable(backward):
        raise TypeError("register_numpy: forward 与 backward 都必须可调用")

    def op(*args, **kwargs):
        tensors = [a for a in args if isinstance(a, Tensor)]
        if not tensors:
            raise TypeError(
                "算子 %r 的输入里没有 Tensor。修法：至少传一个 Tensor（纯 numpy 输入直接算即可）"
                % (name,))
        arrays = [a.data if isinstance(a, Tensor) else a for a in args]
        out_data = np.asarray(forward(*arrays, **kwargs), dtype=np.float32)
        need_grad = any(getattr(a, "requires_grad", False) for a in args)
        out = Tensor(out_data, requires_grad=need_grad, _parents=tuple(tensors))

        def bw(g, grads):
            grads_in = backward(np.asarray(g, np.float32), out_data, arrays, **kwargs)
            if not isinstance(grads_in, (list, tuple)):
                grads_in = [grads_in]
            for a, gi in zip(args, grads_in):
                if isinstance(a, Tensor) and a.requires_grad and gi is not None:
                    _acc(grads, a, np.asarray(gi, dtype=np.float32))

        out._backward = bw
        return out

    return register(name, op, kind=kind, doc=doc, replace=replace)


def unregister(name):
    """卸载插件算子（恢复核心原样，若它覆盖过核心则无法自动恢复 —— 会明确告知）。"""
    rec = _REGISTRY.pop(name, None)
    if rec is None:
        raise KeyError("算子 %r 没有注册过。已注册：%s" % (name, ", ".join(sorted(_REGISTRY)) or "(空)"))
    if rec.get("replaced_core"):
        raise RuntimeError(
            "算子 %r 覆盖过核心实现，卸载无法自动恢复原实现。"
            "修法：重启进程（核心定义在 import 时建立）" % (name,))
    delattr(F, name)
    if rec["kind"] == "tensor" and hasattr(Tensor, name):
        delattr(Tensor, name)
    return rec


def list_ops():
    """列出当前已注册的插件算子（人读格式）。"""
    rows = []
    for name in sorted(_REGISTRY):
        r = _REGISTRY[name]
        rows.append(dict(name=name, kind=r["kind"],
                         has_backward=r["backward"] is not None or r["kind"] == "numpy",
                         replaced_core=r.get("replaced_core", False),
                         doc=(r.get("doc") or "").splitlines()[0] if r.get("doc") else ""))
    return rows


def registered_names():
    return sorted(_REGISTRY)


# ---------------- 发现机制 -------------------------------------------------

def load_plugin_file(path):
    """加载单个 .py 插件文件（模块内调用 register 即可生效）。"""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("插件文件不存在：%s" % (path,))
    mod_name = "tgai_plugin_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_from_env():
    """按 TG_OPS_PATH（os.pathsep 分隔的目录或文件）加载插件。"""
    raw = os.environ.get("TG_OPS_PATH", "").strip()
    if not raw:
        return []
    loaded = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        if os.path.isdir(part):
            for fn in sorted(os.listdir(part)):
                if fn.endswith(".py") and not fn.startswith("_"):
                    loaded.append(load_plugin_file(os.path.join(part, fn)))
        elif part.endswith(".py"):
            loaded.append(load_plugin_file(part))
    return loaded


def load_entry_points(group="tgai_go.ops"):
    """加载已安装包声明的插件入口点（[project.entry-points."tgai_go.ops"]）。"""
    try:
        from importlib.metadata import entry_points
    except Exception:  # noqa: BLE001
        return []
    loaded = []
    try:
        eps = entry_points()
        selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
    except Exception:  # noqa: BLE001
        return []
    for ep in selected:
        try:
            obj = ep.load()
            if callable(obj):
                obj()                 # 约定：入口点是一个「注册函数」
            loaded.append(ep)
        except Exception as exc:  # noqa: BLE001
            # 插件坏了不能拖垮引擎：记录后继续
            sys.stderr.write("[tgai_go.ops] 插件 %s 加载失败：%s\n" % (getattr(ep, "name", "?"), exc))
    return loaded


def autoload():
    """引擎 import 时调用：先按入口点、再按 TG_OPS_PATH 自动加载。"""
    load_entry_points()
    load_from_env()


# 说明：这里**不在 import 时自动 autoload** —— 自动执行用户代码应当是显式的。
# 需要自动加载请在启动脚本里调用 ops_plugin.autoload()，或设 TG_OPS_AUTOLOAD=1。
if os.environ.get("TG_OPS_AUTOLOAD", "").strip() in ("1", "true", "yes"):
    autoload()
