# -*- coding: utf-8 -*-
"""验证分组 GEMM 修复（0.9.32）：

1. **形状一致**时走分组调用，数值必须与逐次 `gemm_nt` 一致（≤1e-5）；
2. **形状不一致**时安全回退到逐次调用，不崩、数值仍正确；
3. 顺带测一下分组 vs 逐次的速度（同形状，如专家行数相同）。

运行（需独占 GPU）：python grouped_fix_check.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")


def log(m):
    print(m, flush=True)


def main():
    import torch
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    ok = True

    def rel(a, b):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))

    # ---------- 1) 形状一致：分组 vs 逐次 ----------
    K, N, M, G = 1024, 4096, 128, 8
    As = [rt.h2d(rng.normal(0, 0.02, (M, K)).astype(np.float32)) for _ in range(G)]
    Bs = [rt.h2d(rng.normal(0, 0.02, (N, K)).astype(np.float32)) for _ in range(G)]
    try:
        got = [np.asarray(rt.d2h(c), np.float32) for c in rt.grouped_nt(As, Bs)]
        ref = [np.asarray(rt.d2h(rt.gemm_nt(a, b)), np.float32) for a, b in zip(As, Bs)]
        worst = max(rel(x, y) for x, y in zip(got, ref))
        log("① 形状一致（M=%d×%d 组）：最大相对差 %.2e ⇒ %s"
            % (M, G, worst, "一致" if worst <= 1e-5 else "**不一致**"))
        ok = ok and worst <= 1e-5
    except Exception as exc:
        log("① 失败（应当不再崩）：%s: %s" % (type(exc).__name__, str(exc)[:160]))
        ok = False

    # ---------- 2) 形状不一致：必须安全回退 ----------
    Ms = [96, 128, 160, 64]
    As2 = [rt.h2d(rng.normal(0, 0.02, (m, K)).astype(np.float32)) for m in Ms]
    Bs2 = [rt.h2d(rng.normal(0, 0.02, (N, K)).astype(np.float32)) for _ in Ms]
    try:
        got2 = [np.asarray(rt.d2h(c), np.float32) for c in rt.grouped_nt(As2, Bs2)]
        ref2 = [np.asarray(rt.d2h(rt.gemm_nt(a, b)), np.float32) for a, b in zip(As2, Bs2)]
        worst2 = max(rel(x, y) for x, y in zip(got2, ref2))
        log("② 形状不一致（M=%s）：最大相对差 %.2e ⇒ %s"
            % (Ms, worst2, "一致（已安全回退）" if worst2 <= 1e-5 else "**不一致**"))
        ok = ok and worst2 <= 1e-5
    except Exception as exc:
        log("② 失败：%s: %s" % (type(exc).__name__, str(exc)[:160]))
        ok = False

    # ---------- 3) 速度：同形状分组 vs 逐次 ----------
    def ev(fn, rep=20):
        fn(); rt.sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(rep):
            fn()
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / rep

    try:
        t1 = ev(lambda: [rt.gemm_nt(a, b) for a, b in zip(As, Bs)])
        t2 = ev(lambda: rt.grouped_nt(As, Bs))
        log("③ 同形状（M=%d×%d 组）：逐次 %.2f ms → 分组 %.2f ms（%.2fx）"
            % (M, G, t1, t2, t1 / max(t2, 1e-9)))
    except Exception as exc:
        log("③ 测速失败：%s: %s" % (type(exc).__name__, str(exc)[:120]))

    log("总结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
