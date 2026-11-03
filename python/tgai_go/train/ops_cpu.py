# -*- coding: utf-8 -*-
"""CPU 后端的 C 插件：用户给 `.c`，引擎调用本机编译器编成动态库并用 ctypes 加载。

## 为什么要它

`ops_plugin` 的 Python 组合在 CPU 上也能跑，但每个算子都要过一遍 numpy 的临时数组，
拿不到循环融合/向量化。给 `.c` 就能写成单趟循环、开 `-O3 -march=native`。

## 编译器从哪来（按顺序找）

1. 环境变量 `TGAI_CC`（显式指定，最稳）
2. `clang` / `gcc` / `cc` / `tcc`（PATH 里的）
3. MSVC 的 `cl.exe`（若在开发者命令提示符里）
4. 找不到 ⇒ **明确报错并给出下载建议**，绝不静默降级

便携编译器（免安装，解压即用）：
* TCC（约 1 MB）：https://bellard.org/tcc/  → 解压后设 `TGAI_CC=D:\\tcc\\tcc.exe`
* w64devkit（gcc 全家桶，约 100 MB）：https://github.com/skeeto/w64devkit/releases

## C 源码约定

```c
void relu_fwd(const float* a, float* out, int n);                       // 一元
void relu_bwd(const float* g, const float* a, float* da, int n);
```
导出符号用 `extern "C"`/纯 C（无 C++ mangling）。引擎按名字取符号。
"""
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile

_CACHE = {}
_COMPILERS = ("clang", "gcc", "cc", "tcc", "cl")


def find_compiler():
    """返回 (可执行路径, 种类) 或 (None, None)。"""
    env = os.environ.get("TGAI_CC")
    if env:
        p = shutil.which(env) or (env if os.path.isfile(env) else None)
        if p:
            return p, ("tcc" if "tcc" in os.path.basename(p).lower() else "unix")
    for name in _COMPILERS:
        p = shutil.which(name)
        if p:
            return p, ("msvc" if name == "cl" else
                       "tcc" if name == "tcc" else "unix")
    # 4) 便携 Zig（pip install ziglang）——零配置可用，且自带 libc 头
    try:
        import ziglang
        z = os.path.join(os.path.dirname(ziglang.__file__), "zig.exe")
        if not os.path.isfile(z):
            z = os.path.join(os.path.dirname(ziglang.__file__), "zig")
        if os.path.isfile(z):
            return z, "zig"
    except Exception:
        pass
    return None, None


def _compile(src_path, out_path, cc, kind):
    if kind == "msvc":
        cmd = [cc, "/nologo", "/O2", "/LD", src_path, "/Fe:" + out_path]
    elif kind == "tcc":
        cmd = [cc, "-shared", "-O2", "-o", out_path, src_path]
    elif kind == "zig":
        cmd = [cc, "cc", "-shared", "-O3", "-o", out_path, src_path]
    else:
        cmd = [cc, "-shared", "-fPIC", "-O3", "-o", out_path, src_path]
        if sys.platform == "win32":
            cmd.insert(1, "-Wl,--out-implib," + out_path + ".a")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError(
            "编译 CPU 插件失败（%s）：\n%s\n修法：确认源码是纯 C（无 C++ 特性）、"
            "导出符号用 extern \"C\"，或用 TGAI_CC 指定编译器"
            % (os.path.basename(cc), ((r.stderr or r.stdout) or "")[:800]))


def load_source(source, name="tgai_cpu_plugin"):
    """编译一份 C 源码并返回句柄（含 lib 与符号缓存）。"""
    cc, kind = find_compiler()
    if cc is None:
        raise RuntimeError(
            "找不到 C 编译器，无法编译 CPU 插件。修法（任一）：\n"
            "  1) 装便携版 TCC（约 1 MB）：https://bellard.org/tcc/ ，解压后设 "
            "TGAI_CC=<路径>/tcc.exe\n"
            "  2) 装 w64devkit（gcc）：https://github.com/skeeto/w64devkit/releases\n"
            "  3) 不装编译器也行：给插件包加 python 档，会自动降级（慢但正确）")
    if os.path.isfile(str(source)):
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    tmpdir = tempfile.mkdtemp(prefix="tgai_cpu_")
    suffix = ".dll" if sys.platform == "win32" else ".so"
    src_path = os.path.join(tmpdir, name + ".c")
    out_path = os.path.join(tmpdir, name + suffix)
    with open(src_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(source)
    _compile(src_path, out_path, cc, kind)
    lib = ctypes.CDLL(out_path)
    _CACHE[name] = {"lib": lib, "path": out_path, "cc": cc, "syms": {}}
    return _CACHE[name]


def _sym(handle, sym):
    s = handle["syms"].get(sym)
    if s is not None:
        return s
    try:
        s = getattr(handle["lib"], sym)
    except AttributeError:
        raise RuntimeError(
            "CPU 插件里找不到符号 %r（%s）。修法：确认函数已定义且不是 static，"
            "C++ 需 extern \"C\"" % (sym, os.path.basename(handle["path"])))
    s.restype = None
    _CACHE[handle["path"]] = handle
    handle["syms"][sym] = s
    return s


def _call(fn, bufs, n):
    """把缓冲的 host 指针按 (void*, void*, ..., int) 调用。

    CPU 张量给的是 numpy 数组（不是 GpuBuf），这里两种都吃；numpy 侧要保证连续，
    并且**在调用期间持有引用**，否则可能被 GC 掉。
    """
    import numpy as _np
    keep = []
    args = []
    for b in bufs:
        if isinstance(b, _np.ndarray):
            arr = _np.ascontiguousarray(b, dtype=_np.float32)
            keep.append(arr)
            args.append(ctypes.c_void_p(int(arr.ctypes.data)))
        else:
            ptr = getattr(b, "host_ptr", None) or getattr(b, "ptr", None)
            args.append(ctypes.c_void_p(int(ptr)))
    args.append(ctypes.c_int(int(n)))
    fn(*args)
    del keep


def register_elementwise(name, handle, fwd, da, db=None, arity=1):
    """把 CPU C 内核接到 Tensor 上（前向/反向都在 host 内存里跑）。"""
    import numpy as np
    import tgai_go.train as T
    from tgai_go.train import ops_plugin

    _sym(handle, fwd)
    _sym(handle, da)
    if db:
        _sym(handle, db)

    def op(*args):
        ts = [T._as_tensor(a) for a in args]
        shape = tuple(ts[0].shape)
        n = int(np.prod(shape)) if shape else 1
        out_data = np.empty(n, dtype=np.float32)
        with np.errstate(all="ignore"):
            _call(_sym(handle, fwd), [t.data for t in ts] + [out_data], n)
        out = T.Tensor(out_data.reshape(shape),
                       requires_grad=any(t.requires_grad for t in ts),
                       _parents=tuple(ts) if any(t.requires_grad for t in ts) else ())

        def bw(g, grads):
            gg = np.asarray(g, dtype=np.float32).reshape(n)
            if ts[0].requires_grad:
                d = np.empty(n, dtype=np.float32)
                _call(_sym(handle, da), [gg] + [t.data for t in ts] + [d], n)
                T._acc(grads, ts[0], d.reshape(shape))
            if arity == 2 and ts[1].requires_grad:
                d = np.empty(n, dtype=np.float32)
                _call(_sym(handle, db or da), [gg] + [t.data for t in ts] + [d], n)
                T._acc(grads, ts[1], d.reshape(shape))

        out._backward = bw
        return out

    ops_plugin.register(name, op)
    return op
