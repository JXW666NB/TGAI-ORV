# -*- coding: utf-8 -*-
"""通用 AMP 基准：典型 dense transformer 的 GEMM 形状上，ORV(fp32/fp16) vs PyTorch(fp32/AMP)。

**为什么不拿 TGAI 的形状说话**：TGAI 的 M=512、K=1024 属于"小 GEMM"，算力根本打不满，
fp16 tensor core 自然没收益。通用库要对标 PyTorch，就得看用户真正常见的形状：
seq 拉长、d_model/d_ff 放大（FFN 的 K/N 是 4096~11008 这种）。

每个形状只测**一次 matmul 的前向**（这是 AMP 收益最直接的地方），best of N。
TG_FP16 在 import 时读取，所以各模式必须独立进程跑。

运行（需独占 GPU）：
    python amp_gemm_bench.py orv_fp32
    python amp_gemm_bench.py orv_fp16      # TG_FP16=1 + 权重影子缓存
    python amp_gemm_bench.py pyt_fp32
    python amp_gemm_bench.py pyt_amp
"""
import os
import sys
import time as _time

import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else "orv_fp32"
if MODE == "orv_fp16":
    os.environ["TG_FP16"] = "1"

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

# (M, K, N) —— 名字只是便于阅读
SHAPES = [
    ("attn qkv   T512  d1024 -> 3072", 512, 1024, 3072),
    ("ffn  up    T512  d1024 -> 4096", 512, 1024, 4096),
    ("attn qkv   T2048 d2048 -> 6144", 2048, 2048, 6144),
    ("ffn  up    T2048 d2048 -> 8192", 2048, 2048, 8192),
    ("ffn  gate  T2048 d4096 -> 11008", 2048, 4096, 11008),
    ("lm head    T2048 d2048 -> 32000", 2048, 2048, 32000),
]
REP = 5


def bench_orv(with_cache):
    import torch
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    if with_cache:
        rt.f16_cache_enable(True)
    for name, M, K, N in SHAPES:
        rng = np.random.default_rng(0)
        a_np = rng.normal(0, 0.1, (M, K)).astype(np.float32)
        b_np = rng.normal(0, 0.1, (K, N)).astype(np.float32)
        a = T.tensor(a_np)
        b = T.tensor(b_np)          # 当作"权重"：同一块内存反复用，正好吃影子缓存

        def once():
            return a @ b

        once(); torch.cuda.synchronize()
        ts = []
        for _ in range(REP):
            torch.cuda.synchronize()
            t0 = _time.time()
            once()
            torch.cuda.synchronize()
            ts.append((_time.time() - t0) * 1000.0)
        log("%-34s %4dx%5dx%-6d | ORV %s %8.2f ms"
            % (name, M, K, N, "fp16" if with_cache else "fp32", min(ts)))
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

        once(); torch.cuda.synchronize()
        ts = []
        for _ in range(REP):
            torch.cuda.synchronize()
            t0 = _time.time()
            once()
            torch.cuda.synchronize()
            ts.append((_time.time() - t0) * 1000.0)
        log("%-34s %4dx%5dx%-6d | PyT %s %8.2f ms"
            % (name, M, K, N, "AMP" if amp else "fp32", min(ts)))
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
