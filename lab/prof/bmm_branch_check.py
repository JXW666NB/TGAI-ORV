# -*- coding: utf-8 -*-
"""分支级验证：matmul 反向的"批量转置 GEMM"映射是否与旧路径逐位一致。

上一轮改成
    da(...,M,k) = g(...,M,n) @ b(...,k,n)^T   ⇒ sgemm_nt(g, b, batch, M, k, n)
    db(...,k,N) = a(...,M,k)^T @ g(...,M,N)   ⇒ sgemm_tn(a, g, batch, k, N, M)
端到端 loss 逐位一致只能证明"模型走到的路径"对；
这里对每个 batch/维度组合单独对拍，包括 batch>1 与 4-D。
"""
import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)


def _batch(shape):
    n = 1
    for s in shape[:-2]:
        n *= int(s)
    return n


def check(case):
    a_shape, b_shape = case
    a = rt.h2d(rng.normal(0, 0.05, a_shape).astype(np.float32))
    b = rt.h2d(rng.normal(0, 0.05, b_shape).astype(np.float32))
    g_shape = a_shape[:-1] + (b_shape[-1],)
    g = rt.h2d(rng.normal(0, 0.05, g_shape).astype(np.float32))

    M_, n_ = g_shape[-2], g_shape[-1]
    k_ = b_shape[-2]

    # ---- da 的两种算法 ----
    new_da = rt.sgemm_nt_strided_batched(g, b, _batch(g_shape), M_, k_, n_)
    old_da = rt.matmul_general(g, rt.swap_last2(b))
    # 生产代码里会 reshape 回 a_shape（批量维被打平成 batch），这里照做
    new_da = rt.reshape(new_da, a_shape) if tuple(new_da.shape) != a_shape else new_da
    d1 = float(np.abs(np.asarray(rt.d2h(new_da)) -
                      np.asarray(rt.d2h(old_da))).max())

    # ---- db 的两种算法 ----
    new_db = rt.sgemm_tn_strided_batched(a, g, _batch(a_shape), k_, n_, M_)
    old_db = rt.matmul_general(rt.swap_last2(a), g)
    new_db = rt.reshape(new_db, b_shape) if tuple(new_db.shape) != b_shape else new_db
    d2 = float(np.abs(np.asarray(rt.d2h(new_db)) -
                      np.asarray(rt.d2h(old_db))).max())

    ok = (tuple(new_da.shape) == tuple(old_da.shape)
          and tuple(new_db.shape) == tuple(old_db.shape)
          and d1 == 0.0 and d2 == 0.0)
    print("  %-22s x %-20s batch=%-3d  da=%.3e db=%.3e  %s"
          % (str(a_shape), str(b_shape), _batch(a_shape), d1, d2,
             "OK" if ok else "**不一致**"))
    return ok


def main():
    print("=== matmul 反向：批量转置 GEMM vs 旧路径（应 max_abs = 0）===")
    cases = [
        ((64, 32), (32, 16)),                 # 2-D
        ((2, 64, 32), (2, 32, 16)),           # 3-D, batch=2
        ((1, 64, 32), (1, 32, 16)),           # 3-D, batch=1
        ((2, 3, 64, 32), (2, 3, 32, 16)),     # 4-D, batch=6
        ((1, 16, 128, 64), (1, 16, 64, 32)),  # attention 形状
        ((2, 512, 1024), (2, 1024, 256)),     # 大 batch
        ((1, 512, 1024), (1, 1024, 512)),     # batch=1 大形状
    ]
    bad = 0
    for c in cases:
        if not check(c):
            bad += 1
    print("\n判定：%s" % ("全部逐位一致" if bad == 0 else "**%d 例不一致**" % bad))
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
