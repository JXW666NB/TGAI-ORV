# -*- coding: utf-8 -*-
"""关键实验：PEDANTIC math mode 是否在压制 tensor core。

## 背景

ORV 初始化时调 `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)`，为的是与 OpenBLAS
逐位一致。**PEDANTIC 会禁用 tensor core**。

V100 的 fp16 tensor core 是 125 TFLOP/s，fp32 只有 15.7。若 MoE 专家 GEMM 能吃到
tensor core，`332 ms → 200 ms` 就有希望。这可能是唯一还没试过的杠杆。

## 测什么

MoE gate 投影的真实形状（8 专家 × [128,1024] @ [4096,1024]）：

| 用例 | 说明 |
| --- | --- |
| fp32 GEMM @ PEDANTIC | 现状基线 |
| fp32 GEMM @ DEFAULT | 数学模式对 fp32 有没有影响 |
| fp16 GEMM @ PEDANTIC | 现状 fp16 |
| **fp16 GEMM @ DEFAULT** | **关键：放开 tensor core** |

运行（需独占 GPU）：python tensorcore_probe.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

D, DF, NE, ROWS = 1024, 4096, 8, 128


def log(m):
    print(m, flush=True)


def main():
    import torch
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)

    def ev(fn, rep=20):
        for _ in range(5):
            fn()
        rt.sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(rep):
            fn()
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / rep

    A32 = [rt.h2d(rng.normal(0, 0.02, (ROWS, D)).astype(np.float32)) for _ in range(NE)]
    W32 = [rt.h2d(rng.normal(0, 0.02, (DF, D)).astype(np.float32)) for _ in range(NE)]
    flops = float(NE * 2 * ROWS * D * DF)

    # fp16 缓冲：用 ORV 自己的转换
    try:
        A16 = [rt.cast_f32_to_f16(x) for x in A32]
        W16 = [rt.cast_f32_to_f16(x) for x in W32]
        have16 = True
    except Exception as exc:
        log("构造 fp16 缓冲失败：%s: %s" % (type(exc).__name__, str(exc)[:120]))
        have16 = False

    has_f16gemm = hasattr(rt, "sgemm_fp16_nt")

    import ctypes
    cb = rt.cublas
    DEFAULT_MATH, PEDANTIC_MATH = 0, 1

    def bench(label, fn):
        t = ev(fn)
        log("%-26s %7.3f ms  （%5.1f TFLOP/s）" % (label, t, flops / (t * 1e9)))
        return t

    # ---- fp32 ----
    cb.cublasSetMathMode(rt.handle, PEDANTIC_MATH)
    t_f32_ped = bench("fp32 @ PEDANTIC", lambda: [rt.gemm_nt(a, w) for a, w in zip(A32, W32)])
    cb.cublasSetMathMode(rt.handle, DEFAULT_MATH)
    t_f32_def = bench("fp32 @ DEFAULT", lambda: [rt.gemm_nt(a, w) for a, w in zip(A32, W32)])

    # ---- fp16 ----
    t_f16_ped = t_f16_def = None
    if have16 and has_f16gemm:
        cb.cublasSetMathMode(rt.handle, PEDANTIC_MATH)
        try:
            t_f16_ped = bench("fp16 @ PEDANTIC（现状）",
                              lambda: [rt.sgemm_fp16_nt(a, w) for a, w in zip(A16, W16)])
        except Exception as exc:
            log("fp16 @ PEDANTIC 失败：%s" % str(exc)[:120])
        cb.cublasSetMathMode(rt.handle, DEFAULT_MATH)
        try:
            t_f16_def = bench("fp16 @ DEFAULT（放开 TC）",
                              lambda: [rt.sgemm_fp16_nt(a, w) for a, w in zip(A16, W16)])
        except Exception as exc:
            log("fp16 @ DEFAULT 失败：%s" % str(exc)[:120])
    else:
        log("（无 fp16 GEMM 接口或 fp16 缓冲，跳过 fp16 用例）")

    cb.cublasSetMathMode(rt.handle, PEDANTIC_MATH)
    log("")
    log("已恢复 PEDANTIC math mode")
    if t_f32_ped and t_f32_def:
        log("fp32: DEFAULT/PEDANTIC = %.2fx" % (t_f32_ped / max(t_f32_def, 1e-9)))
    if t_f16_ped and t_f16_def:
        log("fp16: PEDANTIC→DEFAULT 加速 = %.2fx" % (t_f16_ped / max(t_f16_def, 1e-9)))
        log("若该加速能作用到 MoE：332 ms → 约 %.0f ms（目标 ≤200）" % (332.0 / max(t_f16_ped / max(t_f16_def, 1e-9), 1e-9)))


if __name__ == "__main__":
    main()
