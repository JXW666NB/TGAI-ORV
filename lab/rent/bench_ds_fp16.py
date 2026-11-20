# -*- coding: utf-8 -*-
"""dS（注意力反向的 dL/dscores）走 fp16 + 缩放 的单算子微基准。

口径（全部实测）：
  * 形状 = probe_bmm_sites.py 实测的**真实形状**（b=16 的批量 GEMM，seq2048/dh=64/Tc=512）；
  * 现状 = `cublasSgemmStridedBatched`（fp32）；
    候选 = `cublasGemmStridedBatchedEx`（A/B=fp16，C=fp32，COMPUTE_32F），
    **缩放折进 alpha**（alpha=2^-k，精确的 2 的幂）；
  * 计时用 **CUDA event 包住 N 次背靠背调用**，warmup 后 3 轮取中位，报极差；
  * 误差对 **numpy float64 参考**（逐 batch 元素）。
  * 额外测：cast（fp32→fp16）成本、scale（out-of-place）成本、整张量归约（absmax 代理）成本、
    bf16 在这台 V100 上到底能不能用/多快。

产物：D:\\orv_bench\\bench_ds_fp16.json
跑法（远端）：D:\\orv_bench\\remote_py.bat D:\\orv_bench\\bench_ds_fp16.py
"""
import ctypes
import gc
import json
import os
import statistics
import sys

WORK = r"D:\orv_bench"
OUT = os.path.join(WORK, "bench_ds_fp16.json")
os.chdir(WORK)
sys.path.insert(0, WORK)

import numpy as np                                              # noqa: E402

import tgai_go.train as T                                       # noqa: E402

RT = T._cuda()

CUDA_OP_N, CUDA_OP_T = 0, 1
CUDA_R_16F, CUDA_R_32F, CUDA_R_16BF = 2, 0, 14
CUBLAS_COMPUTE_32F = 68
CUBLAS_GEMM_DEFAULT = -1
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05
FP16_MIN_SUB = 5.960464477539063e-08

_DONE = {"ssb": False, "ssbex": False}


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


def _ev():
    ev = ctypes.c_void_p()
    RT._chk(RT.cu.cuEventCreate(ctypes.byref(ev), ctypes.c_uint(0)), "cuEventCreate")
    return ev


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


def _call_ex(g, C, alpha=1.0, dt=CUDA_R_16F):
    _bind()
    a = ctypes.c_float(float(alpha))
    bb = ctypes.c_float(0.0)
    RT._activate()
    rc = RT.cublas.cublasGemmStridedBatchedEx(
        RT.handle, g[0], g[1], g[2], g[3], g[4],
        ctypes.byref(a),
        ctypes.c_void_p(g[5]), dt, g[6], g[7],
        ctypes.c_void_p(g[8]), dt, g[9], g[10],
        ctypes.byref(bb),
        ctypes.c_void_p(int(C.ptr)), CUDA_R_32F, g[2], g[11],
        g[12], CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
    RT._chk_blas(rc, "cublasGemmStridedBatchedEx")


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


# 真实形状 + 每步次数（来自 probe_bmm_sites.py 的调用点清单）
CASES = [
    dict(tag="sdpa_dq", op="NN", b=16, M=2048, N=64, K=2048, per_step=3, big="A",
         note="dq = dS@k（读 dS 268MB）"),
    dict(tag="sdpa_dk", op="TN", b=16, M=2048, N=64, K=2048, per_step=3, big="A",
         note="dk = dS^T@q（读 dS 268MB）"),
    dict(tag="csa_dq", op="NT", b=16, M=2048, N=64, K=512, per_step=9, big="A",
         note="CSA dq = dS@kT^T（读 dS 67MB）"),
    dict(tag="csa_dk", op="TN", b=16, M=64, N=512, K=2048, per_step=9, big="B",
         note="CSA dk = q^T@dS（读 dS 67MB，B 侧）"),
    dict(tag="csa_dq_loc", op="NT", b=16, M=2048, N=64, K=32, per_step=9, big="A",
         note="局部窗 dq（Lk=32）"),
    dict(tag="csa_dk_loc", op="TN", b=16, M=64, N=32, K=2048, per_step=9, big="B",
         note="局部窗 dk（Lk=32）"),
]


def _big_shape(op, b, M, N, K):
    """被缩放/被 cast 的那个操作数（dS）的数学形状。"""
    if op == "NN":
        return (b, M, K)
    if op == "NT":
        return (b, M, K)
    return (b, K, M)


def _other_shape(op, b, M, N, K):
    if op == "NN":
        return (b, K, N)
    if op == "NT":
        return (b, N, K)
    return (b, K, N)


def run_case(c, rng, ds_absmax, ds_std, kscale):
    op, b, M, N, K = c["op"], c["b"], c["M"], c["N"], c["K"]
    a_shp = {"NN": (b, M, K), "NT": (b, M, K), "TN": (b, K, M)}[op]
    b_shp = {"NN": (b, K, N), "NT": (b, N, K), "TN": (b, K, N)}[op]
    big = c["big"]
    nb = {"A": a_shp, "B": b_shp}[big]
    sm = {"A": a_shp, "B": b_shp}[{"A": "B", "B": "A"}[big]]

    nbig = (rng.standard_normal(nb) * ds_std).astype(np.float32)
    # 加上一个 absmax 的离群点，保证 absmax 与实测一致
    nbig.reshape(-1)[0] = ds_absmax
    nbig.reshape(-1)[1] = -ds_absmax
    nsm = (rng.standard_normal(sm) * 0.64).astype(np.float32)
    nA, nB = (nbig, nsm) if big == "A" else (nsm, nbig)

    A = RT.h2d(nA, dtype=np.float32)
    B = RT.h2d(nB, dtype=np.float32)
    C32 = RT.empty((b, M, N), dtype=np.float32)
    C16 = RT.empty((b, M, N), dtype=np.float32)
    A16 = RT.empty(a_shp, dtype=np.float16)
    B16 = RT.empty(b_shp, dtype=np.float16)
    RT.cast_f32_to_f16_into(A, A16)
    RT.cast_f32_to_f16_into(B, B16)
    RT.sync()

    g = _args(op, b, M, N, K, A, B)
    g16 = _args(op, b, M, N, K, A16, B16)
    big32, big16 = (A, A16) if big == "A" else (B, B16)
    sm32, sm16 = (B, B16) if big == "A" else (A, A16)

    # ---- 误差（前 2 个 batch 元素，fp64 参考）----
    nb_ref = min(b, 2)
    A64 = nA[:nb_ref].astype(np.float64)
    B64 = nB[:nb_ref].astype(np.float64)
    ref = _ref(op, A64, B64)
    _call_fp32(g, C32)
    _call_ex(g16, C16, 1.0)
    RT.sync()
    got32 = RT.d2h(C32)[:nb_ref].astype(np.float64)
    got16 = RT.d2h(C16)[:nb_ref].astype(np.float64)
    e32 = _errs(got32, ref)
    e16 = _errs(got16, ref)

    # ---- 计时 ----
    def f32():
        _call_fp32(g, C32)

    def f16():
        _call_ex(g16, C16, 1.0)

    def f16_cast():
        RT.cast_f32_to_f16_into(big32, big16)
        RT.cast_f32_to_f16_into(sm32, sm16)
        _call_ex(g16, C16, 1.0)

    def f16_scalecast():
        tmp = RT.scale(big32, float(2.0 ** kscale))
        RT.cast_f32_to_f16_into(tmp, big16)
        RT.cast_f32_to_f16_into(sm32, sm16)
        _call_ex(g16, C16, float(2.0 ** (-kscale)))

    def cast_only():
        RT.cast_f32_to_f16_into(big32, big16)

    def scale_only():
        RT.scale(big32, float(2.0 ** kscale))

    def reduce_only():
        RT.sumsq_parallel(big32)

    est = _time(f32, 3)
    n_iter = int(max(20, min(200, round(60.0 / max(est, 1e-3)))))
    t32 = [_time(f32, n_iter) for _ in range(3)]
    t16 = [_time(f16, n_iter) for _ in range(3)]
    tc = [_time(f16_cast, n_iter) for _ in range(3)]
    tsc = [_time(f16_scalecast, n_iter) for _ in range(3)]
    tcast = [_time(cast_only, n_iter) for _ in range(3)]
    tscale = [_time(scale_only, n_iter) for _ in range(3)]
    tred = None
    try:
        tred = statistics.median([_time(reduce_only, n_iter) for _ in range(3)])
    except Exception as e:                                       # noqa: BLE001
        print("   [warn] reduce: %r" % (e,), flush=True)

    # ---- bf16 可用性/速度 ----
    bf = {"ok": False}
    try:
        A16b = RT.cast_f32_to_bf16_flat(A)
        B16b = RT.cast_f32_to_bf16_flat(B)
        gbf = _args(op, b, M, N, K, A16b, B16b)
        Cbf = RT.empty((b, M, N), dtype=np.float32)
        _call_ex(gbf, Cbf, 1.0, CUDA_R_16BF)
        RT.sync()
        gotbf = RT.d2h(Cbf)[:nb_ref].astype(np.float64)
        tbf = statistics.median([_time(lambda: _call_ex(gbf, Cbf, 1.0, CUDA_R_16BF),
                                       max(5, n_iter // 4)) for _ in range(3)])
        bf = {"ok": True, "ms": tbf, "err": _errs(gotbf, ref),
              "tflops": 2.0 * b * M * N * K / 1e9 / tbf}
        del A16b, B16b, Cbf
    except Exception as e:                                       # noqa: BLE001
        bf = {"ok": False, "err": repr(e)}

    out = dict(c)
    out.update({
        "n_iter": n_iter, "kscale": kscale,
        "fp32_ms": statistics.median(t32), "fp32_spread": _spread(t32),
        "fp16_ms": statistics.median(t16), "fp16_spread": _spread(t16),
        "fp16_cast_ms": statistics.median(tc),
        "fp16_scalecast_ms": statistics.median(tsc),
        "cast_only_ms": statistics.median(tcast),
        "scale_only_ms": statistics.median(tscale),
        "reduce_ms": tred,
        "tflops_fp32": 2.0 * b * M * N * K / 1e9 / statistics.median(t32),
        "tflops_fp16": 2.0 * b * M * N * K / 1e9 / statistics.median(t16),
        "gb_read_fp32": nb[0] * nb[1] * nb[2] * 4 / 1e9,
        "gb_read_fp16": nb[0] * nb[1] * nb[2] * 2 / 1e9,
        "err_fp32": e32, "err_fp16": e16, "bf16": bf,
    })
    gc.collect()
    RT.sync()
    return out


def _spread(v):
    m = statistics.median(v)
    return 100.0 * (max(v) - min(v)) / m if m else 0.0


def main():
    rng = np.random.default_rng(12345)
    # 实测 dS 量级（step2 / global layer3）：absmax 2.8e-5, std 9.8e-8
    ds_absmax = float(os.environ.get("DS_ABSMAX", 2.8e-5))
    ds_std = float(os.environ.get("DS_STD", 9.8e-8))
    kscale = int(os.environ.get("DS_K", 30))
    res = []
    for c in CASES:
        print("[case] %s ..." % c["tag"], flush=True)
        try:
            r = run_case(c, rng, ds_absmax, ds_std, kscale)
        except Exception as e:                                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = dict(c)
            r["error"] = repr(e)
        res.append(r)
        print("   fp32 %.4f | fp16 %.4f | +cast %.4f | +scale+cast %.4f | cast %.4f | scale %.4f | red %s"
              % (r.get("fp32_ms", -1), r.get("fp16_ms", -1), r.get("fp16_cast_ms", -1),
                 r.get("fp16_scalecast_ms", -1), r.get("cast_only_ms", -1),
                 r.get("scale_only_ms", -1), r.get("reduce_ms")), flush=True)
        print("   err fp32 relFro=%.2e  fp16 relFro=%.2e  bf16=%r"
              % (r.get("err_fp32", {}).get("rel_fro", -1),
                 r.get("err_fp16", {}).get("rel_fro", -1),
                 (r.get("bf16", {}).get("ok"), r.get("bf16", {}).get("ms"),
                  r.get("bf16", {}).get("err", {}).get("rel_fro")
                  if r.get("bf16", {}).get("ok") else r.get("bf16", {}).get("err"))),
              flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"cases": res, "ds_absmax": ds_absmax, "ds_std": ds_std,
                   "kscale": kscale}, f, ensure_ascii=False, indent=1)
    print("[bench] wrote %s" % OUT)

    print("\n%-12s %5s %6s %8s %8s %8s %9s %8s %10s"
          % ("tag", "n/步", "b", "fp32 ms", "fp16 ms", "+cast", "+scale+cast",
             "reduce", "ms/步 fp32"))
    t32 = t16 = tsc = 0.0
    for r in res:
        if "error" in r or r.get("fp32_ms") is None:
            continue
        n = r["per_step"]
        t32 += n * r["fp32_ms"]
        t16 += n * r["fp16_ms"]
        tsc += n * r["fp16_scalecast_ms"]
        print("%-12s %5d %6d %8.4f %8.4f %8.4f %9.4f %8s %10.3f"
              % (r["tag"], n, r["b"], r["fp32_ms"], r["fp16_ms"],
                 r["fp16_cast_ms"], r["fp16_scalecast_ms"],
                 ("%.4f" % r["reduce_ms"]) if r.get("reduce_ms") else "n/a",
                 n * r["fp32_ms"]))
    print("\n按每步次数加权：fp32 %.2f ms/步 → fp16 %.2f → fp16+scale+cast %.2f ms/步"
          % (t32, t16, tsc))


if __name__ == "__main__":
    main()
