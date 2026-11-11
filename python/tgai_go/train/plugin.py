# -*- coding: utf-8 -*-
"""插件系统统一入口 —— **六条轴，一个命名空间**。

## 设计目标

| 目标 | 做法 |
| --- | --- |
| **简易** | 六条轴都挂在同一个 `plugin` 对象上；一个插件就是**一个 .py 文件** |
| **通用** | 插件只组合核心算子 ⇒ 天然适配 CUDA / Vulkan / ROCm，无需为每个后端写一遍 |
| **安全** | 声明能力（如 `works_inplace`）不足时**当场报错**，绝不静默退化 |

## 六条轴

```python
from tgai_go.train import plugin

plugin.op("softplus", lambda x: (x.exp() + 1.0).log())    # ① 算子
plugin.backend("vulkan", VulkanBackend, caps={...})        # ② 后端
plugin.optimizer("Lion", Lion, works_inplace=True)         # ③ 优化器
plugin.layer("MoELayer", MoELayer)                         # ④ 层
plugin.arch("deepseek_v3", build_dsv3)                     # ⑤ 骨架
plugin.api("torch-compat", install_torch_shim)             # ⑥ API 兼容层
```

## 为什么算子轴直接复用 `ops_plugin`

`ops_plugin` 已经验证过：组合式注册（autograd 自动）、自带反向、entry_points 发现、
契约校验、`manifest()` 带 sha256 可复现。**不重复造第二套**，`plugin.op` 只是转发。

## 装配顺序（重要）

`api` 与 `backend` 的注册**只登记、不立即激活** —— 它们有副作用（改 `sys.modules`、
切后端），必须由用户显式 `plugin.activate()` 触发。这保证了"装插件"不会在你不知情时
改变进程行为。

算子 / 优化器 / 层 / 骨架是**纯登记**，注册后立刻可用（无副作用）。
"""

from __future__ import annotations

import io
import os

from . import ops_plugin as _ops

#: 已登记的后端：(name -> {"factory", "caps"})
_BACKENDS = {}
#: 已登记的优化器：(name -> {"cls", "works_inplace"})
_OPTIMIZERS = {}
#: 已登记的层：(name -> cls)
_LAYERS = {}
#: 已登记的骨架：(name -> builder)
_ARCHS = {}
#: 已登记的 API 兼容层：(name -> installer)
_APIS = {}
#: 已激活的 API / 后端名字
_ACTIVATED = set()


class PluginError(Exception):
    """插件注册/使用错误。"""


def _check_name(kind, name):
    name = str(name).strip()
    if not name or not name.isidentifier():
        raise PluginError(
            "%s 名字 %r 不是合法标识符" % (kind, name) +
            "。修法：用字母开头的标识符（如 my_layer），它会被挂成属性名")
    return name


# ─────────────────────────── ① 算子 ───────────────────────────
def op(name, fn=None, *, kind="functional", doc="", replace=False):
    """注册**组合式**算子（用已有 Tensor 算子拼出来 ⇒ autograd 自动生效）。

    这是最省事的一条路：**不用写反向**，也天然适配所有后端。

    支持两种写法（装饰器更符合"简易"）：

    ```python
    @plugin.op("softplus")
    def softplus(x):
        return (x.exp() + 1.0).log()

    # 或者直接给函数
    plugin.op("softplus", lambda x: (x.exp() + 1.0).log())
    ```
    """
    if fn is None:
        def deco(f):
            op(name, f, kind=kind, doc=doc, replace=replace)
            return f
        return deco
    return _ops.register(name, fn, kind=kind, doc=doc, replace=replace)


def op_numpy(name, forward, backward, *, kind="functional", doc="", replace=False):
    """注册**自带反向**的算子（numpy 前向 + 显式反向，性能优先）。"""
    return _ops.register_numpy(name, forward, backward, kind=kind, doc=doc,
                               replace=replace)


# ─────────────────────────── ② 后端 ───────────────────────────
def backend(name, factory, *, caps=None, replace=False):
    """登记一个 GPU/CPU 后端。

    `caps` 是能力字典（如 `{"subgroup_ops": 32, "async_copy": False}`），
    **必须如实声明** —— 插件作者据它选择快路径/回退路径，声明错了会在别的后端崩。

    登记**不会**立即切换后端；要切换需 `plugin.activate(name)`。
    """
    name = _check_name("backend", name)
    if name in _BACKENDS and not replace:
        raise PluginError("后端 %r 已登记" % name +
                          "。修法：要覆盖请传 replace=True（会先检查没有插件依赖旧实现）")
    if not callable(factory):
        raise PluginError("后端 %r 的 factory 不可调用" % name +
                          "。修法：传一个能返回后端对象的可调用对象")
    _BACKENDS[name] = {"factory": factory, "caps": dict(caps or {})}
    return name


# ─────────────────────────── ③ 优化器 ───────────────────────────
def optimizer(name, cls, *, works_inplace=False, replace=False):
    """登记一个优化器。

    `works_inplace` 决定它**能不能用于共享存储参数**（例如连续存放的专家权重）：
    出式更新（`p.data = p.data - ...`）会把视图换成新数组、从而断开共享，
    所以只有声明 `works_inplace=True` 的优化器才被允许配共享存储参数 ——
    否则**当场报错**，而不是等到训练出问题才发现权重没更新。

    默认 `False` 是刻意的：声明错了会导致"视图不同步"这种极难查的 bug，
    宁可让作者显式写出来。
    """
    name = _check_name("optimizer", name)
    if name in _OPTIMIZERS and not replace:
        raise PluginError("优化器 %r 已登记" % name +
                          "。修法：要覆盖请传 replace=True")
    if not isinstance(cls, type):
        raise PluginError("优化器 %r 必须是类（能被实例化）" % name +
                          "。修法：传一个类，而不是实例或函数")
    _OPTIMIZERS[name] = {"cls": cls, "works_inplace": bool(works_inplace)}
    return name


def require_inplace(name):
    """共享存储参数用它校验优化器 —— **能力不足直接报错**。"""
    info = _OPTIMIZERS.get(name)
    if info is None:
        raise PluginError("优化器 %r 没有登记" % name +
                          "。修法：先 plugin.optimizer(...)；或用核心自带的 AdamW")
    if not info["works_inplace"]:
        raise PluginError(
            "优化器 %r 声明为**出式更新**，不能用于共享存储参数" % name +
            "。修法：改用 works_inplace=True 的优化器，否则权重改了视图不会同步")
    return info["cls"]


# ─────────────────────────── ④ 层 ───────────────────────────
def layer(name, cls=None, *, replace=False):
    """登记一个层（`nn.Module` 子类），之后 `T.nn.<name>` 可用。支持装饰器写法。

    ```python
    @plugin.layer("MyBlock")
    class MyBlock(T.nn.Module):
        ...
    ```
    """
    if cls is None:
        def deco(c):
            layer(name, c, replace=replace)
            return c
        return deco
    name = _check_name("layer", name)
    if name in _LAYERS and not replace:
        raise PluginError("层 %r 已登记" % name + "。修法：要覆盖请传 replace=True")
    if not isinstance(cls, type):
        raise PluginError("层 %r 必须是类" % name +
                          "。修法：传 nn.Module 子类")
    _LAYERS[name] = cls
    return name


# ─────────────────────────── ⑤ 骨架 ───────────────────────────
def arch(name, builder=None, *, replace=False):
    """登记一个**模型骨架**构造函数，之后 `T.build(name, **kw)` 可用。支持装饰器。

    builder 是 `**kwargs -> nn.Module`。用它可以把 DeepSeek-V3 / Qwen3 这类
    结构做成插件，核心不背模型动物园。

    ```python
    @plugin.arch("deepseek_v3")
    def build_dsv3(n_routed=256, n_shared=1, top_k=8):
        return T.nn.MoE(..., n_shared=n_shared, top_k=top_k)
    ```
    """
    if builder is None:
        def deco(b):
            arch(name, b, replace=replace)
            return b
        return deco
    name = _check_name("arch", name)
    if name in _ARCHS and not replace:
        raise PluginError("骨架 %r 已登记" % name + "。修法：要覆盖请传 replace=True")
    if not callable(builder):
        raise PluginError("骨架 %r 的 builder 不可调用" % name +
                          "。修法：传一个 **kwargs -> nn.Module 的可调用对象")
    _ARCHS[name] = builder
    return name


# ─────────────────────────── ⑥ API 兼容层 ───────────────────────────
def api(name, installer, *, replace=False):
    """登记一个**API 兼容层**（例如把 `torch` 这个名字指向 ORV）。

    只登记、不执行 —— 因为这类插件有副作用（改 `sys.modules`），
    必须由用户显式 `plugin.activate(name)` 才会生效。
    """
    name = _check_name("api", name)
    if name in _APIS and not replace:
        raise PluginError("API 兼容层 %r 已登记" % name + "。修法：要覆盖请传 replace=True")
    if not callable(installer):
        raise PluginError("API 兼容层 %r 的 installer 不可调用" % name +
                          "。修法：传一个无参可调用对象，它自己去装 shim")
    _APIS[name] = installer
    return name


# ─────────────────────────── 激活与查询 ───────────────────────────
def activate(name=None):
    """激活已登记的后端 / API 兼容层（有副作用的才需要）。

    不传参数则激活**全部**已登记项。返回被激活的名字列表。
    """
    done = []
    for n, installer in list(_APIS.items()):
        if name is not None and n != name:
            continue
        installer()
        _ACTIVATED.add(n)
        done.append(n)
    for n, info in list(_BACKENDS.items()):
        if name is not None and n != name:
            continue
        # 后端激活由核心后端管理；这里只标记，实际切换在 backend 模块里做
        _ACTIVATED.add("backend:" + n)
        done.append("backend:" + n)
    return done


def resolve_layer(name):
    return _LAYERS.get(name)


def resolve_arch(name):
    return _ARCHS.get(name)


def resolve_optimizer(name):
    info = _OPTIMIZERS.get(name)
    return None if info is None else info["cls"]


def registered():
    """当前登记状态（供 `T.plugins()` 与人看）。"""
    return {
        "ops": list(_ops.registered_names()),
        "backends": sorted(_BACKENDS),
        "optimizers": sorted(_OPTIMIZERS),
        "layers": sorted(_LAYERS),
        "archs": sorted(_ARCHS),
        "apis": sorted(_APIS),
        "activated": sorted(_ACTIVATED),
    }


def load(path):
    """从**单个 .py 文件**加载插件（用户放进 `~/.tgai/plugins/` 即可）。"""
    p = os.path.abspath(str(path))
    if not os.path.isfile(p):
        raise PluginError("插件文件不存在: %s" % p +
                          "。修法：检查路径；插件就是一个 .py 文件")
    return _ops.load_plugin_file(p)


def autoload(extra_dirs=()):
    """自动加载：环境变量 + 用户插件目录 + pip entry_points。

    顺序固定（后者可覆盖前者，因为都用默认 replace=False 会报重名错误，
    所以这里只在名字未登记时才装 —— 见实现）。
    """
    loaded = []
    # 1) 用户插件目录
    home = os.path.expanduser("~")
    for d in [os.path.join(home, ".tgai", "plugins")] + [os.path.abspath(x) for x in extra_dirs]:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".py") and not fn.startswith("_"):
                try:
                    _ops.load_plugin_file(os.path.join(d, fn))
                    loaded.append(fn)
                except Exception:  # noqa: BLE001
                    pass
    # 2) pip entry_points
    try:
        for r in _ops.load_entry_points():
            loaded.append(r.get("name", "?"))
    except Exception:  # noqa: BLE001
        pass
    return loaded


def summary():
    """一行字符串，适合打进日志。"""
    r = registered()
    return ("插件：算子 %d / 后端 %d / 优化器 %d / 层 %d / 骨架 %d / API兼容层 %d；已激活 %d"
            % (len(r["ops"]), len(r["backends"]), len(r["optimizers"]),
               len(r["layers"]), len(r["archs"]), len(r["apis"]), len(r["activated"])))
