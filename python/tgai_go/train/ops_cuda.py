# -*- coding: utf-8 -*-
"""CUDA 算子插件：用户写 .cu 源码（引擎用 NVRTC 编译）或直接给已编译 cubin。

为什么需要：`ops_plugin` 的 A/B 两类插件都是 **Python 组合**，走的是已有算子拼装，
拿不到 C 端的融合收益（多一次读写就多一次带宽）。这里让用户的内核真正跑在 GPU 上。

三条路（按推荐顺序）：
  1. `load_source(src)` —— 用户给 CUDA C 源码字符串**或 .cu 文件路径**，NVRTC 现编
  2. `load_cubin(path)` —— 用户给**已编译**的 cubin/fatbin（离线 nvcc 编好的，含 -lineinfo 都行）
  3. （已有）`ops_plugin.register` —— 纯 Python 组合，无编译、跨后端，但没融合收益

约定（内核签名，n = 元素总数，调用方保证张量连续且元素数相同）：
    unary  fwd: (const float* a, float* out, int n)
           bwd: (const float* g, const float* a, float* da, int n)
    binary fwd: (const float* a, const float* b, float* out, int n)
           bwd: (const float* g, const float* a, const float* b, float* da, int n)
                （db 省略时按 da 的同一内核算，命名给 bwd_b）
用法：
    h = load_source(open("gelu.cu").read())
    register_elementwise("gelu", h, "gelu_fwd", "gelu_bwd")          # 一元
    register_elementwise("swiglu", h, "swiglu_fwd", "swiglu_da", "swiglu_db", arity=2)
"""
import ctypes
import os

import numpy as np

_HANDLES = {}


def _rt():
    import tgai_go.train as T
    return T._cuda()


def _arch(rt):
    """当前卡的 NVRTC 架构串（compute_XY）。用运行时自己的探测，避免手写。"""
    fn = getattr(rt, "_cc_arch", None)
    if fn is not None:
        return fn()
    raise RuntimeError("拿不到当前卡的 compute capability。修法：确认 _cuda() 已初始化")


def load_source(source, cu_name="tgai_plugin.cu", include_paths=None):
    """编译 CUDA C 源码并加载，返回可传给 launch/register 的模块句柄。

    `source` 可以是源码字符串，也可以是 .cu 文件路径。编译失败会带上 NVRTC 日志。
    """
    if os.path.isfile(str(source)):
        with open(source, "r", encoding="utf-8") as f:
            source = f.read()
    rt = _rt()
    rt._activate()
    # 出 cubin 必须给**物理架构** sm_XX（compute_XX 没有 cubin，NVRTC 会报 rc=0 失败）
    sm = _arch(rt).replace("compute_", "sm_")
    cubin = rt._compile(sm, source=source, cu_name=cu_name, want_cubin=True)
    mod = ctypes.c_void_p()
    rc = rt.cu.cuModuleLoadData(ctypes.byref(mod), cubin)
    if rc != 0:
        raise RuntimeError(
            "加载插件模块失败 rc=%d。修法：确认源码里只有 __global__/__device__ 函数"
            "（不要有 host 代码、不要 #include 需要 -I 的头）" % rc)
    _HANDLES[id(mod)] = mod
    return mod


def load_cubin(path):
    """加载**已编译**的 cubin/fatbin（nvcc -cubin 或 -fatbin 产物）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到 cubin：%s。修法：检查路径" % path)
    with open(path, "rb") as f:
        blob = f.read()
    if not blob:
        raise RuntimeError("cubin 是空文件：%s" % path)
    rt = _rt()
    rt._activate()
    mod = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(blob, len(blob))
    rc = rt.cu.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p))
    if rc != 0:
        raise RuntimeError(
            "加载 cubin 失败 rc=%d（%s）。修法：确认这份 cubin 的目标架构与当前卡一致"
            "（sm_70 / sm_86 …），且是同一次 CUDA 大版本编出来的" % (rc, path))
    _HANDLES[id(mod)] = mod
    return mod


def _fn(mod, name):
    key = (id(mod), name)
    cache = _HANDLES.setdefault("_fns", {})
    fn = cache.get(key)
    if fn is not None:
        return fn
    rt = _rt()
    out = ctypes.c_void_p()
    rc = rt.cu.cuModuleGetFunction(ctypes.byref(out), mod, name.encode("ascii"))
    if rc != 0:
        raise RuntimeError(
            "插件里找不到内核 %r（rc=%d）。修法：确认源码里写的是 extern \"C\" __global__ "
            "void %s(...)，或者用不带 C++ mangling 的名字" % (name, rc, name))
    cache[key] = out
    return out


def launch(mod, kernel, grid, block, bufs, shared=0):
    """按 GpuBuf 列表发射插件内核（顺序即内核参数顺序）。"""
    rt = _rt()
    args = [(int(b.ptr), ctypes.c_uint64) for b in bufs]
    rt.launch(kernel, grid, block, args, shared=shared, module=mod)


def register_elementwise(name, module, fwd, da, db=None, arity=1):
    """把插件内核注册成 Tensor 算子（自动接 autograd）。

    arity=1：fwd(a, out, n) / bwd: da(g, a, da, n)
    arity=2：fwd(a, b, out, n) / bwd: da(g, a, b, da, n)；db 给定时同理
    """
    import tgai_go.train as T
    if arity not in (1, 2):
        raise ValueError("arity 只能是 1 或 2，收到 %r" % (arity,))
    _fn(module, fwd)
    for k in ([da] if db is None else [da, db]):
        if k:
            _fn(module, k)

    def op(*args):
        ts = [T._as_tensor(a) for a in args]
        if len(ts) != arity:
            raise ValueError("算子 %s 需要 %d 个输入，收到 %d" % (name, arity, len(ts)))
        rt = _rt()
        shape = tuple(ts[0].shape)
        n = int(np.prod(shape)) if shape else 1
        out = rt.empty(shape, dtype=np.float32)
        grid, block = rt._ewise_grid(n)
        rt.launch(fwd, grid, block,
                  [(int(t._gpu.ptr), ctypes.c_uint64) for t in ts]
                  + [(int(out.ptr), ctypes.c_uint64), (n, ctypes.c_int)],
                  module=module)
        rg = any(t.requires_grad for t in ts)
        out_t = T.Tensor._from_gpu(out, requires_grad=rg, _parents=tuple(ts) if rg else ())

        def bw(g, grads):
            rt2 = _rt()
            n2 = int(np.prod(shape)) if shape else 1
            g2, b2 = rt2._ewise_grid(n2)
            if ts[0].requires_grad:
                d = rt2.empty(shape, dtype=np.float32)
                rt2.launch(da, g2, b2,
                           [(int(g.ptr), ctypes.c_uint64)]
                           + [(int(t._gpu.ptr), ctypes.c_uint64) for t in ts]
                           + [(int(d.ptr), ctypes.c_uint64), (n2, ctypes.c_int)],
                           module=module)
                T._acc(grads, ts[0], d)
            if arity == 2 and ts[1].requires_grad:
                kname = db or da
                d = rt2.empty(shape, dtype=np.float32)
                rt2.launch(kname, g2, b2,
                           [(int(g.ptr), ctypes.c_uint64)]
                           + [(int(t._gpu.ptr), ctypes.c_uint64) for t in ts]
                           + [(int(d.ptr), ctypes.c_uint64), (n2, ctypes.c_int)],
                           module=module)
                T._acc(grads, ts[1], d)

        out_t._backward = bw
        return out_t

    from tgai_go.train import ops_plugin
    ops_plugin.register(name, op)
    return op
