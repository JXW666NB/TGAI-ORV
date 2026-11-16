# -*- coding: utf-8 -*-
"""补测真正的热路径：`sgemm_fp16_nt` / `sgemm_fp16`（fp32 输入，内部自己 cast）。

之前测的 `gemm_f16_*_raw` 是「输入已预 cast」的入口 —— 与模型实际走的
`_sgemm_f16`（内部 `_cast_flat_ptr` / `_cast_f16_cached`）不是同一条路。

重点：
  1. 单个形状的正确性（vs numpy fp64）
  2. **同一权重被反复使用**（模拟 50 步训练里权重指针不变）
  3. **权重内容变化后**（模拟 opt.step 之后）结果是否跟着变
"""
import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)


def rel(got, ref):
    got = np.asarray(got, np.float32).reshape(ref.shape)
    d = float(np.abs(got - ref).max())
    s = max(float(np.abs(ref).max()), 1e-30)
    return d / s


def main():
    print("=== 真热路径 sgemm_fp16_nt（fp32 输入 → 内部 cast）===")
    for (M, K, N) in [(512, 1024, 1024), (512, 1024, 2816), (512, 256, 1024), (256, 1024, 512)]:
        A = rng.normal(0, 1.0, (M, K)).astype(np.float32)
        Bw = rng.normal(0, 0.02, (N, K)).astype(np.float32)
        ref = A.astype(np.float64) @ Bw.astype(np.float64).T
        got = rt.sgemm_fp16_nt(rt.h2d(A), rt.h2d(Bw))
        print("  形状 %-18s rel=%.3e" % (str((M, K, N)), rel(rt.d2h(got), ref)))

    print("\n=== 同一权重反复用 50 次（指针不变，模拟训练步）===")
    M, K, N = 512, 1024, 1024
    Bw = rng.normal(0, 0.02, (N, K)).astype(np.float32)
    Bbuf = rt.h2d(Bw)
    ref = None
    worst = 0.0
    for i in range(50):
        A = rng.normal(0, 1.0, (M, K)).astype(np.float32)
        if ref is None:
            ref = (A.astype(np.float64) @ Bw.astype(np.float64).T)
        got = rt.sgemm_fp16_nt(rt.h2d(A), Bbuf)
        r = rel(rt.d2h(got), A.astype(np.float64) @ Bw.astype(np.float64).T)
        worst = max(worst, r)
    print("  50 次最差 rel=%.3e" % worst)

    print("\n=== 权重内容变了（同一指针覆盖），结果要跟着变 ===")
    A = rng.normal(0, 1.0, (M, K)).astype(np.float32)
    Ab = rt.h2d(A)
    W1 = rng.normal(0, 0.02, (N, K)).astype(np.float32)
    y1 = rt.d2h(rt.sgemm_fp16_nt(Ab, rt.h2d(W1)))
    r1 = rel(y1, A.astype(np.float64) @ W1.astype(np.float64).T)
    W2 = W1 + 0.5                                   # 明显不同的权重
    y2 = rt.d2h(rt.sgemm_fp16_nt(Ab, rt.h2d(W2)))
    r2 = rel(y2, A.astype(np.float64) @ W2.astype(np.float64).T)
    print("  第一次 rel=%.3e   换权重后 rel=%.3e  %s"
          % (r1, r2, "OK" if r2 < 2e-3 else "**BAD（缓存没跟着变）**"))


if __name__ == "__main__":
    main()
