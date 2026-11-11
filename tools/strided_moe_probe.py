# -*- coding: utf-8 -*-
"""决策实验：`cublasSgemmStridedBatched` 在 MoE 上到底能不能赢？

## 为什么先测

分组 GEMM 需要把各专家的行 **pad 到统一 M** 并**把权重打包成连续缓冲**
（strided batched 要求固定 stride）。打包权重的代价正比于权重体积，
而它省下的只是 kernel 启动开销 —— 所以**未必划算**。

- V2.7 配置（8 专家 × 128 行）：每层 24 次 GEMM，启动开销只占 4~8%；
- 真实 MoE（256 专家 × 16 行）：每层 768 次，启动开销才是大头。

本脚本在两种配置下各测一遍，用数据决定要不要实现。

运行（需独占 GPU）：python strided_moe_probe.py
"""
import ctypes
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

D, DF = 1024, 4096


def log(m):
    print(m, flush=True)


def main():
    import torch
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)

    # ---- 1) 符号可用性 ----
    cb = rt.cublas
    for sym in ("cublasSgemmStridedBatched", "cublasSgemmGroupedBatched"):
        try:
            getattr(cb, sym)
            log("符号存在: %s" % sym)
        except AttributeError:
            log("符号缺失: %s" % sym)

    if not hasattr(cb, "cublasSgemmStridedBatched"):
        log("⇒ 无法使用 strided batched，放弃分组方案")
        return

    fn = cb.cublasSgemmStridedBatched
    fn.restype = ctypes.c_int
    # handle, transa, transb, m, n, k, alpha*, A*, lda, strideA, B*, ldb, strideB,
    # beta*, C*, ldc, strideC, batchCount
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
        ctypes.c_int,
    ]
    OP_N, OP_T = 0, 1
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    def ev(f):
        for _ in range(5):
            f()
        rt.sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(20):
            f()
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / 20.0

    def bench(G, M, label):
        K, N = D, DF
        # 每专家 M 行；权重 [N,K]
        A = [rt.h2d(rng.normal(0, 0.02, (M, K)).astype(np.float32)) for _ in range(G)]
        W = [rt.h2d(rng.normal(0, 0.02, (N, K)).astype(np.float32)) for _ in range(G)]

        # ---- 逐专家（现状） ----
        def loop():
            return [rt.gemm_nt(a, w) for a, w in zip(A, W)]

        t_loop = ev(loop)
        flops = float(G * 2 * M * K * N)
        log("  %s：G=%d M=%d → 逐专家 %7.3f ms（%.1f TFLOP/s），%d 次启动"
            % (label, G, M, t_loop, flops / (t_loop * 1e9), G))

        # ---- 打包后的 strided batched ----
        # A 打包成 [G*M, K]，W 打包成 [G*N, K]，C 打包成 [G*M, N]
        # 用 2D 扁平缓冲：cat_place 的 dst_off 是**沿拼接轴的元素偏移**，
        # 所以第 g 个专家放在 g*M / g*N 行处。
        Abuf = rt.empty((G * M, K), dtype=np.float32)
        Wbuf = rt.empty((G * N, K), dtype=np.float32)
        Cbuf = rt.empty((G * M, N), dtype=np.float32)

        def pack():
            for g in range(G):
                rt.cat_place(A[g], Abuf, 0, g * M)
                rt.cat_place(W[g], Wbuf, 0, g * N)

        t_pack = ev(pack)
        log("     打包开销 %7.3f ms（占逐专家的 %.0f%%）" % (t_pack, 100.0 * t_pack / t_loop))

        def strided():
            # C^T = B(T) @ A(N)，与 gemm_nt 同一套列主序
            return fn(rt.handle, OP_T, OP_N, N, M, K,
                      ctypes.byref(alpha),
                      int(Wbuf.ptr), K, int(N * K),
                      int(Abuf.ptr), K, int(M * K),
                      ctypes.byref(beta),
                      int(Cbuf.ptr), N, int(M * N),
                      G)

        try:
            rc = int(strided())
            rt.sync()
        except Exception as exc:
            log("     strided 调用异常：%s" % str(exc)[:120])
            return
        if rc != 0:
            log("     strided 返回 rc=%d，不可用" % rc)
            return
        t_s = ev(strided)
        log("     strided %7.3f ms（%.1f TFLOP/s），1 次启动" % (t_s, flops / (t_s * 1e9)))
        log("     ⇒ 含打包总耗时 %.3f ms，相对逐专家 %.2fx"
            % (t_s + t_pack, (t_s + t_pack) / t_loop))

        # 数值核对（第一组）
        ref = np.asarray(rt.d2h(rt.gemm_nt(A[0], W[0])), np.float32)
        got = np.asarray(rt.d2h(rt.cat_extract(Cbuf, 0, 0, M, (M, N))), np.float32)
        rel = float(np.abs(got - ref).max() / max(1e-9, float(np.abs(ref).max())))
        log("     数值：strided vs gemm_nt 相对差 %.2e ⇒ %s"
            % (rel, "一致" if rel <= 1e-5 else "**不一致**"))

    log("")
    log("配置 A：V2.7 真实配置（8 专家 × 128 行）")
    bench(8, 128, "V2.7")
    log("")
    log("配置 B：真实 MoE 风格（64 专家 × 16 行）")
    bench(64, 16, "大专家数")
    log("")
    log("配置 C：极多专家（256 专家 × 8 行）")
    bench(256, 8, "极多专家")


if __name__ == "__main__":
    main()
