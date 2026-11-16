# -*- coding: utf-8 -*-
"""反向 GEMM 的 transpose 组合对拍：amp 开 vs 关。

签名（读 cuda.py:6076-6115 确认，不猜）：
    gemm_nt(A, B)  A:(M,K) B:(N,K)  ->  C:(M,N) = A @ B^T      【前向 x@W.T】
    gemm_tn(A, B)  A:(K,M) B:(K,N)  ->  C:(M,N) = A^T @ B      【反向 dW = g^T @ x】
    gemm_nn(A, B)  A:(M,K) B:(K,N)  ->  C:(M,N) = A @ B        【反向 dX = g @ W】
"""
import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()


def rel(got, ref):
    got = np.asarray(got, np.float32).reshape(ref.shape)
    return float(np.abs(got - ref).max() / max(float(np.abs(ref).max()), 1e-30))


def run_case(name, M, K, N, amp, seed=11):
    rng = np.random.default_rng(seed)
    if name == "gemm_nt":
        A = rng.normal(0, 1.0, (M, K)).astype(np.float32)
        B = rng.normal(0, 0.02, (N, K)).astype(np.float32)
        ref = A.astype(np.float64) @ B.astype(np.float64).T
    elif name == "gemm_tn":
        A = rng.normal(0, 0.02, (K, M)).astype(np.float32)
        B = rng.normal(0, 1.0, (K, N)).astype(np.float32)
        ref = A.astype(np.float64).T @ B.astype(np.float64)
    else:  # gemm_nn
        A = rng.normal(0, 1.0, (M, K)).astype(np.float32)
        B = rng.normal(0, 0.02, (K, N)).astype(np.float32)
        ref = A.astype(np.float64) @ B.astype(np.float64)
    if amp:
        T.amp_on(dtype="fp16", weight_cache=True)
    else:
        T.amp_off(dtype="fp16")
    try:
        got = rt.d2h(getattr(rt, name)(rt.h2d(A), rt.h2d(B)))
        r = rel(got, ref)
    except Exception as e:
        r = float("nan")
        print("  %-8s (%d,%d,%d) EXC %s: %s" % (name, M, K, N, type(e).__name__, str(e)[:60]))
    T.amp_off(dtype="fp16")
    return r


def main():
    print("=== 反向 GEMM 组合：amp 开 vs 关 ===")
    # 训练里真实出现的量级：T=512 tokens, d=1024, d_ff=2816, latent=256
    cases = [(512, 1024, 1024), (512, 1024, 2816), (1024, 512, 256), (3860, 512, 1024)]
    for M, K, N in cases:
        for name in ("gemm_nt", "gemm_tn", "gemm_nn"):
            r_off = run_case(name, M, K, N, False)
            r_on = run_case(name, M, K, N, True)
            bad = (r_on == r_on) and (r_on > 2e-3)
            print("  %-8s (%4d,%4d,%4d)  off=%.3e  amp=%.3e%s"
                  % (name, M, K, N, r_off, r_on, "   **BAD**" if bad else ""))


if __name__ == "__main__":
    main()
