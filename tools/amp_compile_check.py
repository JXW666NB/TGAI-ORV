# -*- coding: utf-8 -*-
"""AMP 内核的**编译验证**：只让 NVRTC 编译，不分配显存、不碰 GPU 计算。

为什么要单独这个工具：
* 内核是否"能编过"是纯编译问题，不需要 GPU 计算资源 —— 长稳正在占卡时也能做；
* 等真机跑训练才发现语法错，代价太大（bf16 内核一开始就长这样，错了要重排整个 S1）。

用法：
    python tools/amp_compile_check.py                 # 编 引擎源码 + bf16 内核
    python tools/amp_compile_check.py --arch compute_86
    python tools/amp_compile_check.py --no-bf16       # 只编现有引擎源码（对照）
"""
import argparse
import ctypes
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))


def _cuda_module():
    """拿到**模块**本身。

    注意：`tgai_go.train.cuda` 在包命名空间里被替换成了运行时**实例**，
    直接 `from tgai_go.train import cuda` 拿到的是 `_CudaAPI` 对象，
    访问 `_device_source` 会 AttributeError（踩过一次）。
    """
    import importlib
    return importlib.import_module("tgai_go.train.cuda")


def find_nvrtc():
    """找 libnvrtc。优先走仓库自己的搜索逻辑，再退回常见位置。"""
    C = _cuda_module()

    cand = []
    try:
        dirs = C._dedupe_dirs(C._dll_dirs()) if hasattr(C, "_dedupe_dirs") else []
        hit = C._find_file(["libnvrtc.so", "libnvrtc.so.12", "nvrtc64_120_0.dll"], dirs)
        if hit:
            cand.append(hit)
    except Exception:                              # noqa: BLE001
        pass
    import glob
    import site
    for sp in (site.getsitepackages() or []) + [site.getusersitepackages()]:
        cand += glob.glob(os.path.join(sp, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*"))
    env = os.environ.get("TGAI_NVRTC")
    if env:
        cand.insert(0, env)
    for p in cand:
        if p and os.path.isfile(p):
            return p
    raise SystemExit("找不到 libnvrtc；可用 TGAI_NVRTC=/path/to/libnvrtc.so 指定")


def compile_source(source, arch="compute_86", name=b"amp_check.cu"):
    path = find_nvrtc()
    folder = os.path.dirname(path)
    for fn in os.listdir(folder):
        if fn.startswith("libnvrtc-builtins"):
            try:
                ctypes.CDLL(os.path.join(folder, fn), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    lib.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                                       ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_void_p]
    lib.nvrtcCreateProgram.restype = ctypes.c_int
    lib.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.nvrtcCompileProgram.restype = ctypes.c_int
    lib.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetProgramLogSize.restype = ctypes.c_int
    lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nvrtcGetProgramLog.restype = ctypes.c_int
    lib.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetPTXSize.restype = ctypes.c_int
    lib.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nvrtcGetPTX.restype = ctypes.c_int

    opts = ["--gpu-architecture=%s" % arch, "--std=c++14",
            "--device-as-default-execution-space"]
    ob = [o.encode() for o in opts]
    arr = (ctypes.c_char_p * len(ob))(*ob)
    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), source.encode("utf-8"), name,
                                0, None, None)
    if rc != 0:
        return rc, "nvrtcCreateProgram rc=%d" % rc, 0
    rc = lib.nvrtcCompileProgram(prog, len(ob), arr)
    sz = ctypes.c_size_t()
    log = ""
    if lib.nvrtcGetProgramLogSize(prog, ctypes.byref(sz)) == 0 and sz.value > 1:
        buf = ctypes.create_string_buffer(sz.value)
        lib.nvrtcGetProgramLog(prog, buf)
        log = buf.value.decode("utf-8", "replace")
    ptx = 0
    if rc == 0 and lib.nvrtcGetPTXSize(prog, ctypes.byref(sz)) == 0:
        ptx = int(sz.value)
    return rc, log, ptx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="compute_86")
    ap.add_argument("--no-bf16", action="store_true")
    a = ap.parse_args(argv)

    C = _cuda_module()
    from tgai_go.train import amp

    base = C._device_source()
    parts = [("引擎源码", base)]
    if not a.no_bf16:
        parts.append(("bf16 内核", amp.BF16_KERNELS))

    ok = True
    for label, extra in parts:
        src = base if label == "引擎源码" else (base + "\n" + extra)
        rc, log, ptx = compile_source(src, a.arch)
        print("[%s] 源码 %d KB -> rc=%d  PTX %d 字节  %s"
              % (label, len(src) // 1024, rc, ptx, "OK" if rc == 0 else "FAIL"))
        if rc != 0:
            ok = False
            for line in log.strip().split("\n")[:12]:
                print("   " + line)
    print("结论：%s" % ("PASS —— bf16 内核可以编过（未执行，未分配显存）" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
