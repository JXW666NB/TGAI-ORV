# -*- coding: utf-8 -*-
"""fp16 GEMM 入口的独立对拍（老方法：拆模块逐个测）。

入口约定（读 cuda.py:5435-5495 确认，不要凭直觉）：
    gemm_f16_nt_raw(A, B, M, N, K) -> C[(M,N)] = A[(M,K)] @ B[(N,K)]^T
    gemm_f16_nn_raw(A, B, M, N, K) -> C[(M,K)] = A[(M,N)] @ B[(N,K)]
    gemm_f16_tn_raw(A, B, M, N, K) -> C[(N,K)] = A[(M,N)]^T @ B[(M,K)]

参考值一律用 numpy **fp64** 算，再和 fp32/fp16 结果比，
这样可以区分「fp16 舍入」与「算错了」。

用法（V100）：
    D:\\orv_pip_0962\\Scripts\\python.exe fp16_gemm_isolate.py
"""
import numpy as np

import tgai_go.train as T

rt = T._cuda()
T.set_default_device("cuda")
rng = np.random.default_rng(0)

THRESH_FP16 = 2e-3     # fp16 输入 + fp32 累加，相对误差应在此量级内
THRESH_FP32 = 1e-5


def _np(shape, scale):
    return rng.normal(0, scale, shape).astype(np.float32)


def _to_np(t):
    return np.asarray(rt.d2h(t), np.float32)


def _f16(a):
    """fp32 numpy -> fp16 GpuBuf（入口要求 A16/B16 已是 fp16，传 fp32 会读到垃圾）。"""
    return rt.cast_f32_to_f16(rt.h2d(a))


def _cmp(name, got, ref, thresh):
    got = np.asarray(got, np.float32).reshape(ref.shape)
    d = float(np.abs(got - ref).max())
    s = max(float(np.abs(ref).max()), 1e-30)
    r = d / s
    ok = r < thresh
    print("%-46s max_rel=%.3e  %s" % (name, r, "OK" if ok else "**BAD**"))
    return ok


def run_shape(M, N, K, scale):
    """一个形状，三个入口，与 fp64 参考比。"""
    ok_all = True
    # ---- NT ----
    A, B = _np((M, K), scale), _np((N, K), scale)
    ref = (A.astype(np.float64) @ B.astype(np.float64).T)
    got = _to_np(rt.gemm_f16_nt_raw(_f16(A), _f16(B), M, N, K))
    ok_all &= _cmp("NT M=%d N=%d K=%d s=%.2g" % (M, N, K, scale), got, ref, THRESH_FP16)
    # ---- NN ----
    A, B = _np((M, N), scale), _np((N, K), scale)
    ref = (A.astype(np.float64) @ B.astype(np.float64))
    got = _to_np(rt.gemm_f16_nn_raw(_f16(A), _f16(B), M, N, K))
    ok_all &= _cmp("NN M=%d N=%d K=%d s=%.2g" % (M, N, K, scale), got, ref, THRESH_FP16)
    # ---- TN ----
    A, B = _np((M, N), scale), _np((M, K), scale)
    ref = (A.astype(np.float64).T @ B.astype(np.float64))
    got = _to_np(rt.gemm_f16_tn_raw(_f16(A), _f16(B), M, N, K))
    ok_all &= _cmp("TN M=%d N=%d K=%d s=%.2g" % (M, N, K, scale), got, ref, THRESH_FP16)
    return ok_all


def main():
    print("=== fp16 GEMM 入口逐个对拍（参考 = numpy fp64）===")
    bad = []
    # 从小形状到大形状；scale 覆盖真实激活量级（0.02 权重 / 1.0 激活）
    trials = [
        (64, 64, 64, 0.02),
        (512, 2816, 1024, 0.02),      # MoE 专家前向真实形状
        (1024, 4096, 1024, 1.0),      # 激活量级 1.0
        (512, 1024, 8192),            # 长 K（dW 的风险区）
    ]
    for t in trials:
        M, N, K = t[0], t[1], t[2]
        scale = t[3] if len(t) > 3 else 0.02
        try:
            if not run_shape(M, N, K, scale):
                bad.append((M, N, K, scale))
        except Exception as e:
            print("形状 (%d,%d,%d) EXC %s: %s" % (M, N, K, type(e).__name__, str(e)[:80]))
            bad.append((M, N, K, scale))
    print("\n=== 汇总 ===")
    print("全部通过" if not bad else "失败形状: %s" % (bad,))


if __name__ == "__main__":
    main()
