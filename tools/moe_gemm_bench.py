# -*- coding: utf-8 -*-
"""微基准：MoE 每层 8 个小 GEMM vs 1 次分组 GEMM，差多少？

决定"修分组路径"值不值得投入。V2.7 的 MoE 每层每专家约 n/2 行（top-2 of 8）：
seq=512 → 每专家约 128 行。

运行（需独占 GPU）：python moe_gemm_bench.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

D, DF, NE = 1024, 4096, 8
ROWS_PER_EXPERT = [128] * NE


def log(m):
    print(m, flush=True)


def main():
    import torch
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)

    w = [rt.h2d(rng.normal(0, 0.02, (DF, D)).astype(np.float32)) for _ in range(NE)]
    rows = [rt.h2d(rng.normal(0, 1, (r, D)).astype(np.float32)) for r in ROWS_PER_EXPERT]

    def ev(fn, rep=20):
        fn()
        rt.sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(rep):
            fn()
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / rep

    # ① 8 次独立 GEMM（当前路径）
    def loop():
        out = [rt.gemm_nt(r, ww) for r, ww in zip(rows, w)]
        return out

    # ② 1 次分组 GEMM
    def grouped():
        return rt.grouped_nt(rows, w)

    try:
        t1 = ev(loop)
        log("① 8 次独立 GEMM：%.2f ms" % t1)
    except Exception as exc:
        log("① 失败：%s: %s" % (type(exc).__name__, str(exc)[:120]))
        return
    try:
        t2 = ev(grouped)
        log("② 1 次分组 GEMM：%.2f ms" % t2)
        log("   加速比 = %.2fx" % (t1 / max(t2, 1e-9)))
    except Exception as exc:
        log("② 分组 GEMM 不可用/失败：%s: %s" % (type(exc).__name__, str(exc)[:160]))
        return

    # 数值一致性（分组 vs 逐次，必须一致）
    try:
        a = [np.asarray(rt.d2h(x), np.float32) for x in loop()]
        b = [np.asarray(rt.d2h(x), np.float32) for x in grouped()]
        worst = max(float(np.abs(x - y).max() / max(1e-9, float(np.abs(x).max())))
                    for x, y in zip(a, b))
        log("数值：分组 vs 逐次 最大相对差 %.2e ⇒ %s"
            % (worst, "一致" if worst <= 1e-5 else "**不一致**"))
    except Exception as exc:
        log("数值校验失败：%s: %s" % (type(exc).__name__, str(exc)[:120]))


if __name__ == "__main__":
    main()
