# -*- coding: utf-8 -*-
"""最小验证：CUDA Graph 在本环境能否捕获 + 重放 + 逐位一致 + 省 CPU。

不碰训练代码。只有一条 matmul 链 + 逐元素，用来验证：
可用性 → arm → 捕获 → 重放 → 数值比对 → CPU 时间对比。
"""
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)

M, K, N = 512, 1024, 1024


def build():
    A = rt.h2d(rng.normal(0, 0.02, (M, K)).astype(np.float32))
    B = rt.h2d(rng.normal(0, 0.02, (K, N)).astype(np.float32))
    return A, B


def work(A, B, out_holder):
    """一次"前向"：matmul + 两次逐元素（模拟真实链路的多 kernel）。"""
    C = rt.sgemm(A, B)
    D = rt.mul(C, C)
    E = rt.add(D, C)
    out_holder[0] = E
    return E


def cpu_ms(fn, n=30):
    fn()                       # 预热
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()   # 只量 CPU 侧提交耗时（不同步）
    rt.sync()
    return (t1 - t0) / n * 1e3


def main():
    if not rt.graph_available():
        print("graph_available = False，无法验证")
        return
    print("graph_available = True")
    if not rt.graph_arm():
        print("graph_arm 失败")
        return
    print("graph_arm ok")

    A, B = build()
    holder = [None]

    # 1) eager 基线（在 armed 流上跑一遍，让 cuBLAS workspace 先分配好）
    E_eager = work(A, B, holder)
    rt.sync()
    base = np.asarray(rt.d2h(E_eager), np.float32)

    # 2) 预留 arena 并捕获
    need = 64 * 1024 ** 2
    rt.reserve_capture_arena(need)
    ok = rt.graph_begin()
    print("graph_begin =", ok)
    if not ok:
        return
    E_cap = work(A, B, holder)
    ok = rt.graph_end()
    print("graph_end   =", ok, " count =", rt.graph_count())
    if not ok:
        return
    rt.sync()
    cap_first = np.asarray(rt.d2h(holder[0]), np.float32)
    d1 = float(np.abs(cap_first - base).max())

    # 3) 重放若干次，确认结果与首捕一致
    for _ in range(3):
        rt.graph_launch()
    rt.sync()
    replay = np.asarray(rt.d2h(holder[0]), np.float32)
    d2 = float(np.abs(replay - base).max())
    d3 = float(np.abs(replay - cap_first).max())

    # 4) CPU 时间：eager vs 重放
    ms_eager = cpu_ms(lambda: work(A, B, holder))
    ms_graph = cpu_ms(lambda: rt.graph_launch())
    print("\n=== 数值 ===")
    print("  捕获首跑 vs eager : max_abs = %.3e" % d1)
    print("  重放后   vs eager : max_abs = %.3e" % d2)
    print("  重放后   vs 首捕 : max_abs = %.3e" % d3)
    print("\n=== CPU 提交耗时（%d 次平均）===" % 30)
    print("  eager  : %8.3f ms" % ms_eager)
    print("  重放   : %8.3f ms  (%.1fx)" % (ms_graph, ms_eager / max(ms_graph, 1e-9)))


if __name__ == "__main__":
    main()
