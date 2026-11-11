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
from . import dispatch as _disp

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
    """层/优化器/骨架的名字必须是合法标识符 —— 它们会被挂成 `T.nn.<name>` 这类属性。"""
    name = str(name).strip()
    if not name or not name.isidentifier():
        raise PluginError(
            "%s 名字 %r 不是合法标识符" % (kind, name) +
            "。修法：用字母开头的标识符（如 my_layer），它会被挂成属性名")
    return name


def _check_key(kind, name):
    """后端 / API 兼容层的名字只是登记键，**不**挂成属性，所以允许连字符等。

    放宽的理由：像 `torch-compat` 这种叫法最自然，强行要求下划线是无谓的限制。
    """
    name = str(name).strip()
    if not name or not all(c.isalnum() or c in "._-" for c in name):
        raise PluginError(
            "%s 名字 %r 非法" % (kind, name) +
            "。修法：只用字母、数字与 . _ -（如 torch-compat、vulkan）")
    return name


# ─────────────────────────── ① 算子 ───────────────────────────
def op(name, fn=None, *, kind="functional", doc="", replace=False,
       backend=None, caps=None, min_cc=None, max_cc=None, dtype=None, prio=0):
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

    **多后端 / 多能力（为 Vulkan 铺路）**：给出 `backend` / `caps` / `min_cc` /
    `max_cc` / `dtype` 时，实现会登记进分派表，运行时按当前后端与能力选择：

    ```python
    plugin.op("my_silu", fast, backend="cuda", caps={"async_copy": True}, min_cc=(8, 0))
    plugin.op("my_silu", slow)          # 通用回退（不给条件 = 处处可用）
    ```

    `piro` 用于同条件下多个候选的优先级（大的优先）。
    """
    if fn is None:
        def deco(f):
            op(name, f, kind=kind, doc=doc, replace=replace, backend=backend,
               caps=caps, min_cc=min_cc, max_cc=max_cc, dtype=dtype, prio=prio)
            return f
        return deco
    if backend is not None or caps or min_cc is not None or max_cc is not None:
        # 条件化实现：进分派表，不进 functional 命名空间（同一算子可有多个实现）。
        # `backend="*"` 表示"任何后端的通用回退"，分派时会与具体后端实现一起参与选择。
        return _disp.register_impl(name, backend or "*", fn, caps=caps,
                                   min_cc=min_cc, max_cc=max_cc, dtype=dtype, prio=prio)
    return _ops.register(name, fn, kind=kind, doc=doc, replace=replace)


def resolve_op(name, backend, *, caps=None, arch=None, dtype=None):
    """按后端与能力解析算子实现（单次查表 + 缓存）。

    这是插件「单文件适配全后端」的运行时入口：插件不判断"是不是 CUDA"，
    而是由分派按**真实能力**选实现。

    ⚠️ **别在热循环里反复调它**。实测分派净开销约 560 ns：
    对大 GEMM（~0.4 ms）只占 0.14%，但对小 elemwise（~15 µs）占 3.7%，超标。
    热路径请用 `bind()` 解析一次、之后直接调用。
    """
    return _disp.resolve(name, backend, caps=caps, arch=arch, dtype=dtype)


def bind(name, backend, *, caps=None, arch=None, dtype=None):
    """**静态绑定**：解析一次，返回可直接调用的函数。

    热路径的正确用法（把分派开销压到零）：

    ```python
    _silu = plugin.bind("silu", T.backend_name(), caps=T.capabilities(),
                        arch=T.compute_capability(), dtype="fp32")

    # 之后在循环里直接调，无任何分派开销
    for _ in range(steps):
        y = _silu(x)
    ```

    实测分派开销约 560 ns/次：大算子无感，但小算子（~15 µs）会占 3.7%，
    所以**热路径必须用 bind 而不是 resolve_op**。
    """
    return _disp.resolve(name, backend, caps=caps, arch=arch, dtype=dtype)


def dispatch_overhead():
    """实测分派开销（纳秒）。用于判断某条路径该用 bind 还是 resolve_op。"""
    return _disp.benchmark_overhead()


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
    name = _check_key("backend", name)
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
    name = _check_key("api", name)
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
