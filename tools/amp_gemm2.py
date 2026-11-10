# -*- coding: utf-8 -*-
"""通用 AMP 基准（修正版）：用 **CUDA event + 多次迭代** 计时，避免 Windows 15.6 ms 计时粒度。

（上一版用 time.time()，小 GEMM 全读成 0.00 ms —— 那个数据是无效的，此处重测。）

形状覆盖典型 dense transformer（用户自定义模型真会遇到的那些），而不是 TGAI 的 MoE 形状。

运行（需独占 GPU）：
    python amp_gemm2.py orv_fp32 | orv_fp16 | pyt_fp32 | pyt_amp
"""
import os
import sys

import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else "orv_fp32"
if MODE == "orv_fp16":
    os.environ["TG_FP16"] = "1"
    if len(sys.argv) > 2 and sys.argv[2] == "relax":
        # 放开阈值：默认只对 K<=1280 且输出>=262144 的 GEMM 用 fp16，
        # 等于把 tensor core 最有用的大形状全挡回去了。这里放开做大形状实验。
        os.environ["TG_FP16_MAX_K"] = "1000000000"
        os.environ["TG_FP16_MIN_OUTPUT"] = "0"

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

SHAPES = [
    ("qkv  T512  d1024->3072", 512, 1024, 3072),
    ("ffn  T512  d1024->4096", 512, 1024, 4096),
    ("qkv  T2048 d2048->6144", 2048, 2048, 6144),
    ("ffn  T2048 d2048->8192", 2048, 2048, 8192),
    ("gate T2048 d4096->11008", 2048, 4096, 11008),
    ("head T2048 d2048->32000", 2048, 2048, 32000),
]
ITERS = 20
WARM = 3


def _time_events(run_once, iters, warm):
    import torch
    for _ in range(warm):
        run_once()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        run_once()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / iters


def bench_orv(with_cache):
    import torch
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    if with_cache:
        rt.f16_cache_enable(True)
    for name, M, K, N in SHAPES:
        rng = np.random.default_rng(0)
        a = T.tensor(rng.normal(0, 0.1, (M, K)).astype(np.float32))
        b = T.tensor(rng.normal(0, 0.1, (K, N)).astype(np.float32))
        ms = _time_events(lambda: a @ b, ITERS, WARM)
        log("%-26s %5dx%5dx%-6d | ORV %-4s %8.2f ms" % (name, M, K, N,
                                                        "fp16" if with_cache else "fp32", ms))
        del a, b
        rt.empty_cache()


def bench_torch(amp):
    import torch
    for name, M, K, N in SHAPES:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.1
        b = torch.randn(K, N, device="cuda", dtype=torch.float32) * 0.1

        def once():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(amp)):
                return a @ b

        ms = _time_events(once, ITERS, WARM)
        log("%-26s %5dx%5dx%-6d | PyT %-4s %8.2f ms" % (name, M, K, N,
                                                        "AMP" if amp else "fp32", ms))
        del a, b
        torch.cuda.empty_cache()


def main():
    if MODE == "orv_fp32":
        bench_orv(False)
    elif MODE == "orv_fp16":
        bench_orv(True)
    elif MODE == "pyt_fp32":
        bench_torch(False)
    elif MODE == "pyt_amp":
        bench_torch(True)
    else:
        log("未知模式 %r" % MODE)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
