# -*- coding: utf-8 -*-
"""Vulkan 算子：GLSL 源码 → SPIR-V（可选）→ compute dispatch → 接 autograd。

与 `ops_cpu` 对称：CPU 张量的数据是 numpy，这里把它当字节流送进 Vulkan 缓冲，
算完再取回来包成 Tensor。前向/反向各一个 SPIR-V 内核，同一套 dispatch 机制。

## 需要外部 SPIR-V 编译器吗

需要。Vulkan 没有运行时 GLSL JIT，所以：
* 你给 `.spv`（预编译）⇒ 直接可用；
* 你给 `.comp` GLSL ⇒ 需要 `glslang`/`glslc`，用 `TGAI_GLSLANG` 指定路径。
  （本项目实测用的是 Khronos 官方 glslang 16.6.0）

## 内核约定（与 ops_cpu / ops_cuda 一致）

    unary  fwd: binding0=a, binding1=out       bwd: binding0=g, binding1=a, binding2=da
    binary fwd: binding0=a, binding1=b, binding2=out
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

_DEV = None
_SPV_CACHE = {}


def _glslang():
    env = os.environ.get("TGAI_GLSLANG")
    if env and os.path.isfile(env):
        return env
    for name in ("glslang", "glslangValidator", "glslc"):
        p = shutil.which(name)
        if p:
            return p
    return None


def compile_glsl(source, profile="vulkan1.1"):
    """GLSL 计算着色器 → SPIR-V 字节。源码可以是字符串或 .comp 路径。"""
    if os.path.isfile(str(source)):
        with open(source, "r", encoding="utf-8") as f:
            source = f.read()
    exe = _glslang()
    if not exe:
        raise RuntimeError(
            "找不到 SPIR-V 编译器（glslang / glslangValidator / glslc）。修法（任一）：\n"
            "  1) 下载 Khronos glslang 官方包，设 TGAI_GLSLANG=<路径>/glslang.exe\n"
            "  2) 装 Vulkan SDK（自带 glslc）\n"
            "  3) 直接给预编译好的 .spv，跳过编译")
    tmp = tempfile.mkdtemp(prefix="tgai_glsl_")
    src = os.path.join(tmp, "k.comp")
    out = os.path.join(tmp, "k.spv")
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write(source)
    if os.path.basename(exe).startswith("glslc"):
        cmd = [exe, "--target-env=" + profile, "-fshader-stage=compute", src, "-o", out]
    else:
        cmd = [exe, "--target-env", profile, "-V", src, "-o", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError("GLSL 编译失败（%s）：\n%s\n修法：检查 #version 与 layout(binding=N) "
                           "是否与内核约定一致" % (os.path.basename(exe),
                                                ((r.stderr or r.stdout) or "")[:600]))
    with open(out, "rb") as f:
        blob = f.read()
    if len(blob) < 20 or blob[:4] != b"\x03\x02\x23\x07":
        raise RuntimeError("编译产物不是合法 SPIR-V（magic 不对）。修法：确认用的是 -V/--target-env")
    return blob


def spv_of(x):
    """接受 SPIR-V 字节 / .spv 路径 / GLSL 源码 / .comp 路径，返回 SPIR-V 字节。"""
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    p = str(x)
    if os.path.isfile(p):
        if p.endswith(".spv"):
            with open(p, "rb") as f:
                return f.read()
        return compile_glsl(p)
    if "\n" in p or p.lstrip().startswith("#version"):
        return compile_glsl(p)
    raise FileNotFoundError("既不是 SPIR-V 字节、也不是存在的文件：%r" % (x,))


def device():
    """惰性创建全局 VulkanCompute（创建一次约几十毫秒，别在热路径里建）。"""
    global _DEV
    if _DEV is None:
        from tgai_go.train.ops_vk_compute import VulkanCompute
        _DEV = VulkanCompute()
    return _DEV


def available():
    try:
        d = device()
        return d is not None
    except Exception:                                  # noqa: BLE001
        return False


def register_elementwise(name, fwd, da, db=None, arity=1):
    """把 Vulkan 内核注册成 Tensor 算子；fwd/da/db 可以是 SPIR-V 字节、.spv 或 GLSL。"""
    import tgai_go.train as T
    from tgai_go.train import ops_plugin

    kf = spv_of(fwd)
    kd = spv_of(da)
    kdb = spv_of(db) if db else None
    dev = device()
    # 预热：编一次，暴露错误尽早
    probe = np.zeros(64, np.float32)
    dev.run(kf, [probe.tobytes()], 64 * 4, 64)

    def op(*args):
        ts = [T._as_tensor(a) for a in args]
        if len(ts) != arity:
            raise ValueError("算子 %s 需要 %d 个输入，收到 %d" % (name, arity, len(ts)))
        shape = tuple(ts[0].shape)
        n = int(np.prod(shape)) if shape else 1
        ins = [np.ascontiguousarray(t.data, dtype=np.float32).tobytes() for t in ts]
        raw = dev.run(kf, ins, n * 4, n)
        out_data = np.frombuffer(raw, np.float32).copy().reshape(shape or (1,))
        rg = any(t.requires_grad for t in ts)
        out = T.Tensor(out_data, requires_grad=rg, _parents=tuple(ts) if rg else ())

        def bw(g, grads):
            gg = np.ascontiguousarray(np.asarray(g, dtype=np.float32)).tobytes()
            if ts[0].requires_grad:
                d = np.frombuffer(dev.run(kd, [gg] + ins, n * 4, n), np.float32).copy()
                T._acc(grads, ts[0], d.reshape(shape or (1,)))
            if arity == 2 and ts[1].requires_grad:
                d = np.frombuffer(dev.run(kdb or kd, [gg] + ins, n * 4, n), np.float32).copy()
                T._acc(grads, ts[1], d.reshape(shape or (1,)))

        out._backward = bw
        return out

    ops_plugin.register(name, op)
    return op


import atexit as _atexit


@_atexit.register
def _close_device():
    """进程退出前销毁 Vulkan 对象；否则驱动 teardown 可能段错误。"""
    global _DEV
    d, _DEV = _DEV, None
    if d is not None:
        try:
            d.close()
        except Exception:
            pass
