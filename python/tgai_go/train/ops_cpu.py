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
            # ★ 2026-11-20：TGAI_CC 指向 zig 时必须走 zig 分支。
            #   原来只认 tcc，其它一律当 "unix" ⇒ 会调用
            #   `zig.exe -shared ...`（缺了 `cc` 子命令）而失败。
            _b = os.path.basename(p).lower()
            return p, ("tcc" if "tcc" in _b else
                       "zig" if "zig" in _b else "unix")
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
    """编译 C 源码为共享库。失败抛 RuntimeError（带编译器原始 stderr）。

    ★ 2026-11-20 修 CI（windows-latest 长期是红的）：
      通用分支原来**无条件**加 `-fPIC`，还插一个 MinGW 专有的
      `-Wl,--out-implib,<out>.a`。GitHub Actions 的 Windows 镜像里 `cc` 是 clang，
      默认目标 `x86_64-pc-windows-msvc`，于是直接报
          clang: error: unsupported option '-fPIC' for target 'x86_64-pc-windows-msvc'
      三个 CPU 插件用例全挂（Linux 腿不受影响，所以只有 Windows 红）。

      事实依据：Windows 的 PE/DLL **不需要** `-fPIC`（代码本身位置无关）；
      `--out-implib` 只为"链接期生成导入库"服务，ctypes 运行时加载用不到。
      所以改成"平台优先 + 失败回退"：Windows 先不加 `-fPIC`，不成就再试一次加上的
      （覆盖 MinGW/gcc 这类要求它的工具链）；非 Windows 保持原来的 `-fPIC`。
    """
    def _try(cmds):
        last = ""
        for cmd in cmds:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and os.path.isfile(out_path):
                return True, ""
            last = ((r.stderr or r.stdout) or "")[:800]
        return False, last

    if kind == "msvc":
        cmds = [[cc, "/nologo", "/O2", "/LD", src_path, "/Fe:" + out_path]]
    elif kind == "tcc":
        cmds = [[cc, "-shared", "-O2", "-o", out_path, src_path]]
    elif kind == "zig":
        cmds = [[cc, "cc", "-shared", "-O3", "-o", out_path, src_path]]
    elif sys.platform == "win32":
        # Windows 上还有第二层坑：**默认不导出任何符号**（Linux 默认全导出）。
        # 若工具链是 clang→MSVC 目标，产物是"没有导出表"的 DLL，ctypes 必然找不到函数。
        # 所以依次尝试：普通 → 显式全导出（MinGW/clang-GNU 认这个）→ 加 -fPIC。
        cmds = [[cc, "-shared", "-O3", "-o", out_path, src_path],
                [cc, "-shared", "-O3", "-Wl,--export-all-symbols",
                 "-o", out_path, src_path],
                [cc, "-shared", "-fPIC", "-O3", "-Wl,--export-all-symbols",
                 "-o", out_path, src_path],
                [cc, "-shared", "-fPIC", "-O3", "-o", out_path, src_path]]
    else:
        cmds = [[cc, "-shared", "-fPIC", "-O3", "-o", out_path, src_path]]

    ok, err = _try(cmds)
    if not ok:
        raise RuntimeError(
            "编译 CPU 插件失败（%s）：\n%s\n修法：确认源码是纯 C（无 C++ 特性）、"
            "导出符号用 extern \"C\"，或用 TGAI_CC 指定编译器"
            % (os.path.basename(cc), err))


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
        _hint = ""
        if sys.platform == "win32":
            # Windows 的第二层坑：DLL 默认**没有导出表**（clang→MSVC 目标尤其如此），
            # 编译成功但 ctypes 找不到函数。这一条要单独讲清楚，否则用户会去改 C 代码。
            _hint = (
                "\n  ⚠️ Windows 特有：编译成功了却找不到符号，通常是**编译器默认不导出符号**。"
                "\n     修法（任一）："
                "\n       1) 用 zig（零配置，推荐）：pip install ziglang，然后设 "
                "TGAI_CC=<...>\\zig.exe"
                "\n       2) 用 MinGW/gcc，或给函数加 __declspec(dllexport)"
                "\n       3) 该工具链不支持全导出时，本模块会依次尝试 "
                "-Wl,--export-all-symbols，仍失败就换编译器")
        raise RuntimeError(
            "CPU 插件里找不到符号 %r（%s）。修法：确认函数已定义且不是 static，"
            "C++ 需 extern \"C\"%s" % (sym, os.path.basename(handle["path"]), _hint))
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
