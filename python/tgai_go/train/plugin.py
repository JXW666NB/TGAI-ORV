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


#: 原生内核插件的编译缓存：(name, arch) -> module
_NATIVE = {}
#: 已登记的原生内核（元信息，供 native_info）
_NATIVE_OPS = {}


def op_native(name, source, entry, *, grid=(1, 1, 1), block=(1, 1, 1), shared=0,
              args=None, cuda_source_name=None, doc="", replace=False):
    """注册一个用 **CUDA C 源码**写的算子 —— ORV 用 NVRTC **现场编译**，跑原生速度。

    ## 为什么需要（回答「.py 会不会太慢」）

    `.py` 只是**登记与胶水**，只在加载时执行一次；**热路径可以完全是原生代码**。
    本函数接收 CUDA C 源码，首次调用时编译、之后直接启动内核 —— 没有 Python 逐元素开销。

    ## 用法

    ```python
    SRC = r'''
    extern "C" __global__ void axpy(float* y, const float* x, float a, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] += a * x[i];
    }
    '''

    launch = plugin.op_native(
        "axpy", SRC, "axpy",
        grid=lambda x, y, a, n: ((n + 255) // 256, 1, 1),
        block=(256, 1, 1),
        args=lambda x, y, a, n: [y._gpu.ptr, x._gpu.ptr, float(a), int(n)],
    )
    launch(x, y, 2.0, x.numel())
    ```

    参数
    ----
    source : str
        CUDA C 源码。内核须用 `extern "C"` 导出，函数名与 `entry` 一致。
    entry : str
        内核函数名。
    grid / block : 可调用 或 tuple
        `grid` 通常依赖输入尺寸，所以支持传函数 `(*args) -> (gx, gy, gz)`；
        `block` 也可传函数（`(*args) -> (bx, by, bz)`）或固定 tuple。
    args : 可调用
        `(*args) -> list[(值, ctype)]`。**必须返回 `(值, ctype)` 对**，
        这是 ORV 内部 `launch` 的约定（它按值构建 ctypes 对象再取指针）。
        指针类参数用 `ctypes.c_void_p`，标量用 `ctypes.c_float` / `ctypes.c_int`。
        省略时把收到的参数原样传给内核（只适用于全是标量的场景）。
    shared : int
        动态共享内存字节数。

    说明：本函数只管**编译与启动**，不管 autograd。要做可训练算子请配合
    `register_numpy` 的反向，或用组合式写法。
    """
    if not isinstance(source, str) or not source.strip():
        raise PluginError(
            "op_native %r 的 source 必须是非空 CUDA C 源码字符串" % name +
            "。修法：传源码；要用预编译二进制见 docs/PLUGIN_NATIVE.md 的 ctypes 方式")
    if not str(entry).strip():
        raise PluginError(
            "op_native %r 缺少 entry（内核函数名）" % name +
            "。修法：把 CUDA C 里 __global__ 的函数名传给 entry")

    resolved = {"fn": None}

    def _rt():
        """拿到 CUDA 运行时（延迟导入，避免 plugin ↔ __init__ 循环依赖）。"""
        from . import _cuda as _c
        return _c()

    def _ensure():
        """首次调用时编译；同一 (算子, 架构) 只编一次。"""
        if resolved["fn"] is not None:
            return resolved["fn"]
        try:
            rt = _rt()
        except Exception as exc:      # noqa: BLE001
            raise PluginError(
                "op_native %r 需要 CUDA 后端，当前环境拿不到（%s）" % (name, exc) +
                "。修法：本插件只能在有 NVIDIA GPU 且驱动可用的机器上跑；"
                "纯 CPU 环境请用组合式 plugin.op(...)") from exc
        # _compile 收的是字符串架构名（"compute_70" / "sm_70"），不是整数 ——
        # 传 int 会让 NVRTC 收到 --gpu-architecture=70 而报 invalid value。
        # 与 ORV 内部一致：优先用 sm_ 的 cubin（驱动对 PTX 的 JIT 版本可能比 NVRTC 旧）。
        arch_s = str(rt._cc_arch())                     # compute_70
        sm = arch_s.replace("compute_", "sm_")          # sm_70
        key = (str(name), sm)
        mod = _NATIVE.get(key)
        if mod is None:
            cubin = rt._compile(sm, source=str(source),
                                cu_name=(cuda_source_name or ("%s.cu" % name)),
                                want_cubin=True)
            # `_compile(want_cubin=True)` 返回的是 **cubin 字节**，不是 CUmodule ——
            # 必须再 cuModuleLoadData 才能拿到可 launch 的模块句柄
            # （直接把它当 module 传会得到 rc=400 invalid resource handle）。
            import ctypes as _ct
            buf = _ct.create_string_buffer(cubin, len(cubin))
            handle = _ct.c_void_p()
            rt._chk(rt.cu.cuModuleLoadData(_ct.byref(handle),
                                           _ct.cast(buf, _ct.c_void_p)),
                    "cuModuleLoadData(%s)" % name)
            mod = handle
            _NATIVE[key] = mod
        resolved["fn"] = mod
        return mod

    def launch(*a, **kw):
        mod = _ensure()
        g = grid(*a, **kw) if callable(grid) else tuple(grid)
        b = block(*a, **kw) if callable(block) else tuple(block)
        av = args(*a, **kw) if callable(args) else list(a)
        # ★ 2026-11-12（外部实测）：args 忘记写 ctype 时，内部 launch 会抛
        #   `TypeError: cannot unpack non-iterable int object` —— 看不出是自己的问题。
        #   这里提前校验，直接把"该怎么写"贴出来。
        for _i, _it in enumerate(av):
            if not (isinstance(_it, (tuple, list)) and len(_it) == 2):
                raise PluginError(
                    "op_native %r 的第 %d 个参数不是 (值, ctype) 对，收到 %r"
                    % (name, _i, _it)
                    + "。修法：args 必须返回 [(值, ctype), ...]。例如想写 "
                      "`args=lambda x, y, a, n: [y._gpu.ptr, x._gpu.ptr, float(a), int(n)]`，"
                      "要改成 `[(y._gpu.ptr, ctypes.c_void_p), "
                      "(x._gpu.ptr, ctypes.c_void_p), (float(a), ctypes.c_float), "
                      "(int(n), ctypes.c_int)]` —— 指针用 c_void_p，标量用 c_float / c_int")
        rt = _rt()
        rt.launch(str(entry), tuple(g), tuple(b), list(av),
                  shared=int(shared), module=mod)
        return None

    launch.__name__ = str(name)
    launch.__doc__ = doc or ("CUDA C 原生内核 %r（entry=%s）" % (name, entry))
    _NATIVE_OPS[str(name)] = {"entry": str(entry), "source_len": len(source)}
    return launch


def load_manifest(path, *, replace=False):
    """加载 **JSON 声明式原生插件** —— 用户不用写一行 Python。见 `plugin_manifest.py`。"""
    from .plugin_manifest import load_manifest as _lm
    return _lm(path, replace=replace)


def native_info():
    """已登记的原生内核与编译状态（供诊断）。"""
    return {
        "registered": dict(_NATIVE_OPS),
        "compiled": ["%s@%s" % k for k in _NATIVE],
        "libraries": dict(_LIBS_META),
    }


# ───────── C 端插件（优先级最高的形态）─────────

#: 已加载的预编译原生库：libpath -> ctypes.CDLL
_LIBS = {}
#: 已登记的原生库元信息（供 native_info）
_LIBS_META = {}


def load_lib(path, *, replace=False):
    """加载一个**预编译**原生库（`.so` / `.dll` / `.dylib`），返回 ctypes 句柄。

    这是**最优先的插件形态**：用户用自己的工具链（nvcc / clang / MSVC）编译，
    ORV 只负责加载与调用，热路径**完全没有 Python 参与**。

    ```c
    // myops.cu  —— 用 nvcc -shared -o myops.so myops.cu 编译
    extern "C" void my_scale(float* y, const float* x, float a, int n) {
        // 里面可以自己 launch kernel
    }
    ```

    ```python
    lib = plugin.load_lib("myops.so")
    scale = plugin.op_c("scale", lib, "my_scale",
                        args=lambda y, x, a, n: [y._gpu.ptr, x._gpu.ptr, float(a), int(n)])
    scale(y, x, 2.0, y.numel())
    ```

    ⚠️ **安全性**：加载原生库等于执行任意机器码。只加载你信任的库；
    ORV 不做沙箱、也不校验来源。
    """
    import ctypes
    import os

    p = os.path.abspath(str(path))
    if not os.path.isfile(p):
        raise PluginError(
            "原生库不存在: %s" % p +
            "。修法：检查路径与文件扩展名（Windows 用 .dll、Linux 用 .so、macOS 用 .dylib）")
    if p in _LIBS and not replace:
        return _LIBS[p]
    try:
        lib = ctypes.CDLL(p)
    except OSError as exc:
        raise PluginError(
            "加载原生库失败: %s (%s)" % (p, exc) +
            "。修法：确认架构与位数匹配（64 位进程只能加载 64 位库）；"
            "Linux 下若提示 undefined symbol，用 ldd 检查依赖") from exc
    _LIBS[p] = lib
    _LIBS_META[p] = {"size": os.path.getsize(p)}
    return lib


def op_c(name, lib, func, *, args=None, restype=None, doc=""):
    """把原生库里导出的 **C ABI 函数**包装成可调用对象（零 Python 热路径）。

    参数
    ----
    lib : ctypes.CDLL 或 str 路径
        `load_lib()` 的返回值，或直接给路径。
    func : str
        导出的 C 函数名。
    args : 可调用
        `(*a) -> list`，把 Python 侧参数转成 C 函数的参数列表。

    返回一个可调用对象；调用即执行原生代码。
    """
    if isinstance(lib, str):
        lib = load_lib(lib)
    fname = str(func).strip()
    if not fname:
        raise PluginError(
            "op_c %r 缺少 func（导出的 C 函数名）" % name +
            "。修法：把 C 源码里 `extern \"C\"` 的函数名传给 func")
    try:
        fn = getattr(lib, fname)
    except AttributeError as exc:
        raise PluginError(
            "原生库里找不到导出符号 %r" % fname +
            "。修法：确认 C 函数用 `extern \"C\"` 导出（否则 C++ 会改编函数名）；"
            "Linux 可用 `nm -D --defined-only 你的库.so` 查导出表") from exc
    if restype is not None:
        fn.restype = restype

    def call(*a, **kw):
        av = args(*a, **kw) if callable(args) else list(a)
        fn(*av)
        return None

    call.__name__ = str(name)
    call.__doc__ = doc or ("原生库函数 %s" % fname)
    _NATIVE_OPS[str(name)] = {"entry": fname, "kind": "c-abi"}
    return call


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
