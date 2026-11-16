# -*- coding: utf-8 -*-
"""把模型里实际出现的 15 种 GEMM 规格逐个重放，fp32 vs fp16 对拍。

形状推导严格按 cuBLAS 文档（ld 是「列主序下的行数」）：
    opA=N ⇒ A_cm 形状 (lda, k)，逻辑 op(A) = A_cm[:m, :]
    opA=T ⇒ A_cm 形状 (lda, m)，逻辑 op(A) = A_cm[:k, :].T
    opB=N ⇒ B_cm 形状 (ldb, n)，逻辑 op(B) = B_cm[:k, :]
    opB=T ⇒ B_cm 形状 (ldb, k)，逻辑 op(B) = B_cm[:n, :].T
    C_cm 形状 (ldc, n)，结果 = C_cm[:m, :]
参考值用 fp64 算 op(A) @ op(B)，与 fp32 / fp16 结果比。
"""
import numpy as np

import tgai_go.train as T
import sys

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)
MOD = sys.modules[CLS.__module__]

SPECS = [
    (1, 0, 2816, 512, 1024, 1024, 1024),
    (0, 0, 1024, 512, 2816, 1024, 2816),
    (0, 1, 1024, 2816, 512, 1024, 2816),
    (0, 0, 1024, 512, 1024, 1024, 1024),
    (1, 0, 1024, 512, 2816, 2816, 2816),
    (0, 0, 2816, 512, 1024, 2816, 1024),
    (0, 1, 2816, 1024, 512, 2816, 1024),
    (0, 0, 1024, 1024, 512, 1024, 512),
    (0, 0, 1024, 256, 128, 1024, 128),
    (0, 0, 1024, 512, 2, 1024, 2),
    (0, 0, 1024, 512, 256, 1024, 256),
    (0, 0, 256, 1024, 512, 256, 512),
    (0, 0, 1024, 512, 96000, 1024, 96000),
    (0, 1, 1024, 96000, 512, 1024, 96000),
    (1, 0, 96000, 512, 1024, 96000, 1024),
]


def build(ta, tb, m, n, k, lda, ldb, seed=5):
    rng = np.random.default_rng(seed)
    a_shape = (lda, k) if ta == 0 else (lda, m)
    b_shape = (ldb, n) if tb == 0 else (ldb, k)
    A = rng.normal(0, 1.0, a_shape).astype(np.float32)
    B = rng.normal(0, 0.02, b_shape).astype(np.float32)
    Aop = A[:m, :] if ta == 0 else A[:k, :].T
    Bop = B[:k, :] if tb == 0 else B[:n, :].T
    ref = Aop.astype(np.float64) @ Bop.astype(np.float64)      # (m,n)
    return A, B, ref


def one(amp, spec):
    ta, tb, m, n, k, lda, ldb = spec
    A, B, ref = build(*spec)
    C = rt.empty((ldc := n, m), dtype=np.float32)             # 列主序 (ldc, n)==(n, m)
    if amp:
        T.amp_on(dtype="fp16", weight_cache=True)
    else:
        T.amp_off(dtype="fp16")
    try:
        rt._sgemm(ta, tb, m, n, k, rt.h2d(A), lda, rt.h2d(B), ldb, C, ldc)
        rt.sync()
        got = np.asarray(rt.d2h(C), np.float32).T            # → (m,n)
    except Exception as e:
        T.amp_off(dtype="fp16")
        return float("nan"), str(e)[:60]
    T.amp_off(dtype="fp16")
    return float(np.abs(got - ref).max() / max(float(np.abs(ref).max()), 1e-30)), ""


def main():
    print("=== 逐个重放模型里的 15 种 GEMM 规格（fp32 vs fp16）===")
    bad = []
    for spec in SPECS:
        r_off, e1 = one(False, spec)
        r_on, e2 = one(True, spec)
        ta, tb, m, n, k, lda, ldb = spec
        is_bad = (r_on == r_on) and (r_on > 2e-3)
        print("  t=%d,%d m=%6d n=%6d k=%6d lda=%6d ldb=%6d  off=%8.2e amp=%8.2e %s%s"
              % (ta, tb, m, n, k, lda, ldb, r_off, r_on,
                 "**BAD**" if is_bad else "", (" EXC:" + e2) if e2 else ""))
        if is_bad:
            bad.append(spec)
    print("\n坏规格:", bad if bad else "无")


if __name__ == "__main__":
    main()
