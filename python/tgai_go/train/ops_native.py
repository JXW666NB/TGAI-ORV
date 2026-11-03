# -*- coding: utf-8 -*-
"""多后端原生算子注册：同一个算子可以有多个"方言"实现，按本机可用性自动挑。

## 为什么需要

`ops_cuda` 只解决 CUDA。但我们要接 ROCm / Vulkan / CPU / 国产卡，
如果插件接口写死 `.cu`，这些卡的用户就没有原生算子了。PyTorch 的做法是按后端分目录
（`aten/src/ATen/native/{cuda,cpu,vulkan}`），这里做成**一份清单 + 自动降级**：

```python
from tgai_go.train import ops_native

ops_native.register("gelu", {
    "cuda":   "gelu.cu",          # NVRTC 现编（NVIDIA）
    "hip":    "gelu.cu",          # hipRTC：CUDA 源码基本可复用（海光 DCU 走这条）
    "musa":   "gelu.cu",          # 摩尔线程 mcc：CUDA 兼容
    "vulkan": "gelu.spv",         # 预编译 SPIR-V（Vulkan 没有标准的运行时 GLSL JIT）
    "cpu":    "gelu.c",           # 小型 C JIT（拿向量化）
    "python": lambda x: ...,      # 永远可用的兜底（慢但正确）
}, arity=1)
```

选取顺序 `cuda -> hip -> musa -> vulkan -> cpu -> python`；缺哪档就往下掉，
**最终一定有实现**。`ops_native.status()` 可查每个算子实际选了哪个后端。

## 各卡的实际可行性（诚实版）

| 后端 | 源码能否复用 CUDA `.cu` | 说明 |
| --- | --- | --- |
| CUDA | — | 原生 |
| HIP（AMD / 海光 DCU） | **≈是** | `__global__`/`threadIdx` 等同形，编译用 hipRTC；少数 intrinsic 需替换 |
| MUSA（摩尔线程） | **≈是** | mcc 兼容 CUDA 语法，通常只换编译器 |
| Vulkan | **否** | GLSL/SPIR-V，只能离线编译后加载 |
| CPU | 否 | 另写 C（或直接给 numpy/Python 组合）|
| 昇腾 Ascend / 寒武纪 | **否** | Ascend C / BANG 是另一套语言与工具链，必须各写一份 |
"""
import ctypes
import os

from tgai_go.train import ops_cuda, ops_plugin

_NATIVE = {}


def _backend_chain():
    """本机确实可用的后端顺序（探测失败就跳过，不抛异常）。"""
    import tgai_go.train as T
    chain = []
    try:
        rt = T._cuda()
        if getattr(rt, "cu", None) is not None:
            chain.append("cuda")
    except Exception:                                  # noqa: BLE001
        pass
    for name, soname in (("hip", "libamdhip64.so"), ("musa", "libmusart.so")):
        try:
            ctypes.CDLL(soname)
            chain.append(name)
        except OSError:
            pass
    chain.append("python")
    return chain


def supported():
    """本机可用的后端列表（用于诊断/文档）。"""
    return _backend_chain()


def register(name, impls, arity=1):
    """按后端方言注册原生算子；没有本机可用方言时退到 python 实现。"""
    if not impls:
        raise ValueError("register(%r) 没有任何实现。修法：至少给一个 python 实现做兜底"
                         % (name,))
    _NATIVE[name] = dict(impls)
    _NATIVE[name]["arity"] = int(arity)
    chain = _backend_chain()
    chosen = None
    for b in chain:
        if b == "python":
            break
        if b in impls:
            chosen = b
            break
    if chosen is None:
        chosen = "python" if "python" in impls else None
    if chosen is None:
        raise ValueError(
            "算子 %r 有实现 %s，但本机没有匹配的后端（可用：%s）。"
            "修法：补一个 python 实现做兜底" % (name, sorted(k for k in impls), chain))
    _NATIVE[name]["chosen"] = chosen
    _NATIVE[name]["chain"] = chain

    if chosen == "cuda":
        src = impls["cuda"]
        cu = os.path.basename(src) if os.path.isfile(src) else "plugin.cu"
        if os.path.isfile(src):
            with open(src, "r", encoding="utf-8") as f:
                src = f.read()
        mod = ops_cuda.load_source(src, cu_name=cu)
        fwd = impls.get("cuda_fwd") or (name + "_fwd")
        da = impls.get("cuda_bwd") or (name + "_bwd")
        db = impls.get("cuda_bwd2")
        return ops_cuda.register_elementwise(name, mod, fwd, da, db, arity=arity)

    fn = impls[chosen]
    if not callable(fn):
        raise ValueError(
            "后端 %r 的实现不是可调用对象（%r）。修法：非 CUDA 后端请直接给 Python 函数"
            % (chosen, type(fn).__name__))
    ops_plugin.register(name, fn)
    return fn


def status():
    """每个已注册算子选了哪个后端（含候选链）。"""
    return {k: {"chosen": v.get("chosen"), "chain": v.get("chain"),
                "impls": sorted(kk for kk in v if kk not in ("arity", "chosen", "chain"))}
            for k, v in _NATIVE.items()}
