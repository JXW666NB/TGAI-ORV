# -*- coding: utf-8 -*-
"""3-D 批量 GEMM 微基准：fp32（现状）vs fp16 输入+fp32 累加（候选）。

口径（全部实测，不是估算）：
  * **真实形状**取自 probe_bmm_sites.py 的每步调用点清单（369 次/步、718.8 GFLOP/步）；
  * 直接 ctypes 调 cuBLAS：`cublasSgemmStridedBatched`（现状）、
    `cublasGemmStridedBatchedEx`（候选，A/B=CUDA_R_16F，C=CUDA_R_32F，
    computeType=CUBLAS_COMPUTE_32F）、`cublasGemmEx`（现有 2-D 路径，交叉参考）；
  * 计时用 **CUDA event 包住 N 次背靠背调用**（不是每次前后 sync —— 那会把小
    kernel 放大几十倍），warmup 后测 3 轮取中位，同时报极差；
  * 误差对 **numpy float64 参考**：逐元素最大绝对误差、Frobenius 相对误差、
    逐元素最大相对误差（分母 max(|ref|,1e-3*absmax)）。

跑法（远端）：D:\\orv_pip_0962\\Scripts\\python.exe -X utf8 D:\\orv_bench\\bench_bmm_fp16.py
"""
import ctypes
import gc
import json
import os
import statistics
import sys

WORK = r"D:\orv_bench"
OUT = os.path.join(WORK, "bench_bmm_fp16.json")
os.chdir(WORK)
sys.path.insert(0, WORK)

import numpy as np                                              # noqa: E402

import tgai_go.train as T                                       # noqa: E402

RT = T._cuda()

# ---- cuBLAS 常量（与 cuda.py 一致）-----------------------------------
CUDA_OP_N, CUDA_OP_T = 0, 1
CUDA_R_16F, CUDA_R_32F = 2, 0
CUBLAS_COMPUTE_32F = 68
CUBLAS_GEMM_DEFAULT = -1
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05
FP16_MIN_SUB = 5.960464477539063e-08

_DONE = {"ssb": False, "ssbex": False, "ex": False}


def _bind():
    cb = RT.cublas
    if not _DONE["ssb"]:
        cb.cublasSgemmStridedBatched.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
            ctypes.c_int]
        cb.cublasSgemmStridedBatched.restype = ctypes.c_int
        _DONE["ssb"] = True
    if not _DONE["ssbex"]:
        cb.cublasGemmStridedBatchedEx.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        cb.cublasGemmStridedBatchedEx.restype = ctypes.c_int
        _DONE["ssbex"] = True
    if not _DONE["ex"]:
        RT._bind_gemm_ex()
        _DONE["ex"] = True


def _ev():
    ev = ctypes.c_void_p()
    RT._chk(RT.cu.cuEventCreate(ctypes.byref(ev), ctypes.c_uint(0)), "cuEventCreate")
    return ev


DBG = bool(os.environ.get("TG_BENCH_DEBUG"))
_TAG = [""]


def _hard(tag):
    """显式检查上下文错误：把「异步炸在后面某个 case」变成「炸在当下这一行」。"""
    _TAG[0] = tag
    rc = int(RT.cu.cuCtxSynchronize())
    if rc != 0:
        raise RuntimeError("cuCtxSynchronize rc=%d @ %s" % (rc, tag))
    if DBG:
        print("      [ok] %s" % tag, flush=True)


def _rec(ev):
    if RT.cu.cuEventRecord(ev, RT._stream) != 0:
        raise RuntimeError("cuEventRecord failed")


def _time(fn, n_iter):
    for _ in range(max(2, n_iter // 10)):
        fn()
    RT.sync()
    e0, e1 = _ev(), _ev()
    _rec(e0)
    for _ in range(n_iter):
        fn()
    _rec(e1)
    RT.sync()
    return RT.event_elapsed_ms(e0, e1) / float(n_iter)


def _args(op, b, M, N, K, A, B):
    """按 ORV cuda.py 的换算给 cuBLAS 实参（行优先语义）。

    返回 (opA, opB, m, n, k, pF, lda, strideF, pS, ldb, strideS, strideC, batch)
      NN: C[M,N]=A[M,K]@B[K,N]   -> opA=N,opB=N
      NT: C[M,N]=A[M,K]@B[N,K]^T -> opA=T,opB=N
      TN: C[M,N]=A[K,M]^T@B[K,N] -> opA=N,opB=T
    """
    if op == "NN":
        return (CUDA_OP_N, CUDA_OP_N, N, M, K,
                int(B.ptr), N, K * N, int(A.ptr), K, M * K, M * N, b)
    if op == "NT":
        return (CUDA_OP_T, CUDA_OP_N, N, M, K,
                int(B.ptr), K, N * K, int(A.ptr), K, M * K, M * N, b)
    if op == "TN":
        return (CUDA_OP_N, CUDA_OP_T, N, M, K,
                int(B.ptr), N, K * N, int(A.ptr), M, K * M, M * N, b)
    raise ValueError(op)


def _call_fp32(g, C):
    _bind()
    a = ctypes.c_float(1.0)
    bb = ctypes.c_float(0.0)
    RT._activate()
    rc = RT.cublas.cublasSgemmStridedBatched(
        RT.handle, g[0], g[1], g[2], g[3], g[4],
        ctypes.byref(a),
        ctypes.c_void_p(g[5]), g[6], g[7],
        ctypes.c_void_p(g[8]), g[9], g[10],
        ctypes.byref(bb),
        ctypes.c_void_p(int(C.ptr)), g[2], g[11], g[12])
    RT._chk_blas(rc, "cublasSgemmStridedBatched")


def _call_fp16(g, C, alpha=1.0):
    """候选：A/B=fp16、C=fp32、fp32 累加（CUBLAS_COMPUTE_32F）。"""
    _bind()
    a = ctypes.c_float(float(alpha))
    bb = ctypes.c_float(0.0)
    RT._activate()
    rc = RT.cublas.cublasGemmStridedBatchedEx(
        RT.handle, g[0], g[1], g[2], g[3], g[4],
        ctypes.byref(a),
        ctypes.c_void_p(g[5]), CUDA_R_16F, g[6], g[7],
        ctypes.c_void_p(g[8]), CUDA_R_16F, g[9], g[10],
        ctypes.byref(bb),
        ctypes.c_void_p(int(C.ptr)), CUDA_R_32F, g[2], g[11],
        g[12], CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
    RT._chk_blas(rc, "cublasGemmStridedBatchedEx")


def _mk(kind, shape, rng):
    n = 1
    for s in shape:
        n *= int(s)
    if kind == "act":
        a = np.clip(rng.standard_normal(n), -5, 5)
    elif kind == "w":
        a = np.clip(rng.standard_normal(n) * 0.02, -0.1, 0.1)
    elif kind == "qk":
        a = np.clip(rng.standard_normal(n) * 0.64, -3.6, 3.6)
    elif kind == "v":
        a = np.clip(rng.standard_normal(n) * 0.32, -1.8, 1.8)
    elif kind == "p_flat":
        a = np.full(n, 1.0 / float(shape[-1]), dtype=np.float64)
    elif kind == "p_peak":
        z = rng.standard_normal(int(shape[-1])) * 2.0
        p = np.exp(z - z.max())
        p /= p.sum()
        a = np.tile(p, n // int(shape[-1]))
    elif kind == "ds":
        a = rng.standard_normal(n) * 4.5e-8
    elif kind == "rope":
        a = rng.uniform(-1, 1, n)
    else:
        raise ValueError(kind)
    return a.astype(np.float32).reshape(shape)


def _ref(op, A64, B64):
    if op == "NN":
        return A64 @ B64
    if op == "NT":
        return A64 @ np.swapaxes(B64, -1, -2)
    return np.swapaxes(A64, -1, -2) @ B64


def _errs(got, ref):
    d = got.astype(np.float64) - ref
    aa = float(np.abs(ref).max()) if ref.size else 0.0
    den = np.maximum(np.abs(ref), max(1e-3 * aa, 1e-30))
    return {"max_abs": float(np.abs(d).max()),
            "rel_fro": float(np.sqrt(np.mean(d * d)) /
                             max(np.sqrt(np.mean(ref * ref)), 1e-300)),
            "max_rel": float((np.abs(d) / den).max()),
            "absmax_ref": aa}


def run_case(c, rng):
    op, b, M, N, K = c["op"], c["b"], c["M"], c["N"], c["K"]
    ak, bk = c["a_kind"], c["b_kind"]
    # 数学层缓冲形状
    a_shp = {"NN": (b, M, K), "NT": (b, M, K), "TN": (b, K, M)}[op]
    b_shp = {"NN": (b, K, N), "NT": (b, N, K), "TN": (b, K, N)}[op]
    if c.get("a2d"):
        a_shp = a_shp[1:]
    if c.get("b2d"):
        b_shp = b_shp[1:]
    nA = _mk(ak, a_shp, rng)
    nB = _mk(bk, b_shp, rng)
    A = RT.h2d(nA, dtype=np.float32)
    B = RT.h2d(nB, dtype=np.float32)
    C32 = RT.empty((b, M, N), dtype=np.float32)
    C16 = RT.empty((b, M, N), dtype=np.float32)
    A16 = RT.empty(a_shp, dtype=np.float16)
    B16 = RT.empty(b_shp, dtype=np.float16)
    RT.cast_f32_to_f16_into(A, A16)
    RT.cast_f32_to_f16_into(B, B16)
    RT.sync()
    _hard("alloc+cast")

    g = _args(op, b, M, N, K, A, B)
    g16 = _args(op, b, M, N, K, A16, B16)

    # ---- 参考与误差 ----
    nb_ref = min(b, 2)
    A64 = nA[:nb_ref].astype(np.float64)
    B64 = nB[:nb_ref].astype(np.float64)
    ref = _ref(op, A64, B64)
    _call_fp32(g, C32)
    _call_fp16(g16, C16)
    RT.sync()
    _hard("first calls")
    got32 = RT.d2h(C32)[:nb_ref].astype(np.float64)
    got16 = RT.d2h(C16)[:nb_ref].astype(np.float64)
    _hard("after d2h")
    e32 = _errs(got32, ref)
    e16 = _errs(got16, ref)
    # fp16 量化本身的误差（只 cast 不 GEMM，对 A 做一次往返）
    q16 = RT.d2h(A16).astype(np.float64).ravel()
    q32 = nA.astype(np.float64).ravel()
    dq = np.abs(q16 - q32)
    frac_sub = float(np.mean(np.abs(q32) < FP16_MIN_NORMAL))
    frac_zero = float(np.mean(np.abs(q32) < FP16_MIN_SUB))
    quant = {"a_absmax": float(np.abs(q32).max()),
             "a_frac_below_fp16_min_normal": frac_sub,
             "a_frac_below_fp16_min_subnormal": frac_zero,
             "a_cast_max_abs_err": float(dq.max()),
             "a_cast_rel_fro": float(np.sqrt(np.mean((q16 - q32) ** 2)) /
                                     max(np.sqrt(np.mean(q32 ** 2)), 1e-300))}

    # ---- 计时 ----
    def f32():
        _call_fp32(g, C32)

    def f16():
        _call_fp16(g16, C16)

    def f16_cast():
        RT.cast_f32_to_f16_into(A, A16)
        RT.cast_f32_to_f16_into(B, B16)
        _call_fp16(g16, C16)

    est = _time(f32, 3)
    n_iter = int(max(20, min(300, round(60.0 / max(est, 1e-3)))))
    t32 = [_time(f32, n_iter) for _ in range(3)]
    _hard("time fp32")
    t16 = [_time(f16, n_iter) for _ in range(3)]
    _hard("time fp16")
    tc = [_time(f16_cast, n_iter) for _ in range(3)]
    _hard("time fp16+cast")
    # cast 单独
    tcast = [_time(lambda: (RT.cast_f32_to_f16_into(A, A16),
                            RT.cast_f32_to_f16_into(B, B16)), n_iter)
             for _ in range(3)]
    _hard("time cast")

    # ---- 交叉参考：cublasGemmEx 逐 batch 循环（现有 2-D 路径）----
    t_ex = None
    if b <= 8:
        _bind()
        a = ctypes.c_float(1.0)
        bb = ctypes.c_float(0.0)

        def loop_ex():
            # 现有 2-D fp16 路径（cublasGemmEx）的等价形式：同一次数学，逐 batch 发 b 次
            RT._activate()
            for i in range(b):
                RT._chk_blas(RT.cublas.cublasGemmEx(
                    RT.handle, g16[0], g16[1], g16[2], g16[3], g16[4],
                    ctypes.byref(a),
                    ctypes.c_uint64(g16[5] + i * g16[7] * 2), CUDA_R_16F, g16[6],
                    ctypes.c_uint64(g16[8] + i * g16[10] * 2), CUDA_R_16F, g16[9],
                    ctypes.byref(bb),
                    ctypes.c_uint64(int(C16.ptr) + i * g16[11] * 4), CUDA_R_32F,
                    g16[2], CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemmEx")

        try:
            tt = [_time(loop_ex, max(5, n_iter // 2)) for _ in range(3)]
            t_ex = statistics.median(tt)
        except Exception as e:                                   # noqa: BLE001
            print("   [warn] gemmEx loop: %r" % (e,), flush=True)
            t_ex = None

    out = dict(c)
    out.update({
        "n_iter": n_iter,
        "fp32_ms": statistics.median(t32), "fp32_spread": _spread(t32),
        "fp16_ms": statistics.median(t16), "fp16_spread": _spread(t16),
        "fp16_cast_ms": statistics.median(tc), "fp16_cast_spread": _spread(tc),
        "cast_only_ms": statistics.median(tcast),
        "gemmex_loop_ms": t_ex,
        "tflops_fp32": 2.0 * b * M * N * K / 1e9 / statistics.median(t32),
        "tflops_fp16": 2.0 * b * M * N * K / 1e9 / statistics.median(t16),
        "err_fp32": e32, "err_fp16": e16, "quant": quant,
    })
    for v in (A, B, C32, C16, A16, B16):
        del v
    gc.collect()
    RT.sync()
    return out


def _spread(v):
    m = statistics.median(v)
    return 100.0 * (max(v) - min(v)) / m if m else 0.0


CASES = [
    # ---- A. 3-D @ 2-D 线性投影（batch=1，正向；matmul_general lead=1）----
    dict(tag="proj_fwd_x_wq", group="A_proj", op="NN", b=1, M=2048, N=1024, K=1024,
         a_kind="act", b_kind="w", per_step=60, note="x[B,T,d] @ wq^T；12 层×5 投影"),
    dict(tag="proj_wkv", group="A_proj", op="NN", b=1, M=2048, N=256, K=1024,
         a_kind="act", b_kind="w", per_step=9, note="CSA 层 kv 低秩下投影"),
    dict(tag="proj_wo", group="A_proj", op="NN", b=1, M=2048, N=1024, K=256,
         a_kind="act", b_kind="w", per_step=9, note="CSA 层 out 投影"),
    dict(tag="proj_kv_tc", group="A_proj", op="NN", b=1, M=512, N=1024, K=256,
         a_kind="act", b_kind="w", per_step=18, note="CSA 层 hc @ wk/wv"),
    # ---- B. dW（权重梯度，TN，batch=1；源码明确记录过精度不可降）----
    dict(tag="dW_main", group="B_dW", op="TN", b=1, M=1024, N=1024, K=2048,
         a_kind="act", b_kind="act", per_step=30, note="x^T @ dq（1024×1024 投影）"),
    dict(tag="dW_kv", group="B_dW", op="TN", b=1, M=1024, N=256, K=2048,
         a_kind="act", b_kind="act", per_step=9, note="wo 的 dW（K=2048）"),
    dict(tag="dW_kv2", group="B_dW", op="TN", b=1, M=256, N=1024, K=512,
         a_kind="act", b_kind="act", per_step=18, note="CSA 层 wk/wv 的 dW"),
    # ---- C. SDPA（3 个 global 层）6 个 GEMM ----
    dict(tag="sdpa_scores", group="C_sdpa", op="NT", b=16, M=2048, N=2048, K=64,
         a_kind="qk", b_kind="qk", per_step=3, note="q@k^T（输出 S×S=268MB）"),
    dict(tag="sdpa_pv", group="C_sdpa", op="NN", b=16, M=2048, N=64, K=2048,
         a_kind="p_flat", b_kind="v", per_step=3, note="p@v（读 P 268MB）"),
    dict(tag="sdpa_dp", group="C_sdpa", op="NN", b=16, M=2048, N=2048, K=64,
         a_kind="v", b_kind="qk", per_step=3, note="dP = g@v^T（写 268MB）"),
    dict(tag="sdpa_dq", group="C_sdpa", op="NN", b=16, M=2048, N=64, K=2048,
         a_kind="ds", b_kind="qk", per_step=3, note="dq = dS@k（读 dS 268MB）"),
    dict(tag="sdpa_dk", group="C_sdpa", op="TN", b=16, M=2048, N=64, K=2048,
         a_kind="ds", b_kind="qk", per_step=3, note="dk = dS^T@q"),
    dict(tag="sdpa_dv", group="C_sdpa", op="NN", b=16, M=2048, N=64, K=2048,
         a_kind="p_flat", b_kind="v", per_step=3, note="dv = p^T@g（读 P 268MB）"),
    # ---- D. CSA（9 层）6 个 GEMM，S×512 ----
    dict(tag="csa_scores", group="D_csa", op="NN", b=16, M=2048, N=512, K=64,
         a_kind="qk", b_kind="qk", per_step=9, note="q@k^T（Lk=512，输出 67MB）"),
    dict(tag="csa_pv", group="D_csa", op="NN", b=16, M=2048, N=64, K=512,
         a_kind="p_flat", b_kind="v", per_step=9, note="p@v"),
    dict(tag="csa_dp", group="D_csa", op="NT", b=16, M=2048, N=512, K=64,
         a_kind="v", b_kind="qk", per_step=9, note="dP = g@v^T"),
    dict(tag="csa_dq", group="D_csa", op="NT", b=16, M=2048, N=64, K=512,
         a_kind="ds", b_kind="qk", per_step=9, note="dq = dS@kT"),
    dict(tag="csa_dk", group="D_csa", op="TN", b=16, M=64, N=512, K=2048,
         a_kind="qk", b_kind="ds", per_step=9, note="dk = q^T@dS（大张量在 B 侧）"),
    dict(tag="csa_dv", group="D_csa", op="TN", b=16, M=512, N=64, K=2048,
         a_kind="p_flat", b_kind="v", per_step=9, note="dv = p^T@g"),
    # ---- E. RoPE（x @ perm，小）----
    dict(tag="rope", group="E_rope", op="NN", b=16, M=2048, N=64, K=64,
         a_kind="act", b_kind="rope", per_step=36, note="RoPE 的 x@perm"),
]


def main():
    rng = np.random.default_rng(12345)
    only = set(sys.argv[1:]) or None
    res = []
    for c in CASES:
        if only and c["tag"] not in only:
            continue
        print("[case] %s ..." % c["tag"], flush=True)
        try:
            r = run_case(c, rng)
        except Exception as e:                                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = dict(c)
            r["error"] = repr(e)
        res.append(r)
        print("   fp32 %.4f ms  fp16 %.4f ms  fp16+cast %.4f ms  cast %.4f"
              % (r.get("fp32_ms", -1), r.get("fp16_ms", -1),
                 r.get("fp16_cast_ms", -1), r.get("cast_only_ms", -1)), flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[bench] wrote %s" % OUT)

    print("\n%-14s %6s %5s %8s %8s %8s %9s %9s %10s"
          % ("tag", "n/步", "b", "fp32 ms", "fp16 ms", "+cast ms",
             "TF32", "TF16", "ms/步 fp32"))
    tot32 = tot16 = tot16c = 0.0
    for r in res:
        if r.get("fp32_ms") is None or "error" in r:
            continue
        n = r["per_step"]
        tot32 += n * r["fp32_ms"]
        tot16 += n * r["fp16_ms"]
        tot16c += n * r["fp16_cast_ms"]
        print("%-14s %6d %5d %8.4f %8.4f %8.4f %9.1f %9.1f %10.3f"
              % (r["tag"], n, r["b"], r["fp32_ms"], r["fp16_ms"],
                 r["fp16_cast_ms"], r["tflops_fp32"], r["tflops_fp16"],
                 n * r["fp32_ms"]))
    print("\n按每步次数加权：fp32 %.2f ms/步 → fp16 %.2f ms/步 → fp16+cast %.2f ms/步"
          % (tot32, tot16, tot16c))
    print("（对照：profiler 实测 fp32 strided-batched 合计 72.6 ms/步）")


if __name__ == "__main__":
    main()
