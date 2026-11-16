# -*- coding: utf-8 -*-
"""每步总 GEMM FLOPs 与步时的相关性 —— 检验「MoE 路由负载不均导致步时波动」。

确定性已经确认（同参数两次运行步时逐位相同），CSA 分块已否证（Tc 恒 128）。
若步时随每步 GEMM 总量变化，就说明是路由把 token 分到各专家的数量不同，
导致瘦长 GEMM 的 n 变化（n=131 这类形状效率极低）。
"""
import argparse
import sys

import numpy as np

from tgai_go.train import v3_pretrain as P

STEPS = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--d_ff", type=int, default=3328)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seq", type=int, default=512)
    a = ap.parse_args()

    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    CLS = type(rt)

    orig = CLS._sgemm
    CUR = {"flops": 0.0, "calls": 0, "ns": []}

    def _sgemm(self, transa, transb, m, n, k, *ar, **kw):
        CUR["flops"] += 2.0 * int(m) * int(n) * int(k)
        CUR["calls"] += 1
        if len(CUR["ns"]) < 400:
            CUR["ns"].append(int(n))
        return orig(self, transa, transb, m, n, k, *ar, **kw)

    CLS._sgemm = _sgemm

    # 每步结束（opt.step 之前）把计数落一条；用 AdamW.step 作为步边界
    from tgai_go.train import optim as _opt_mod
    cur_opt = None
    for nm in dir(_opt_mod):
        v = getattr(_opt_mod, nm)
        if isinstance(v, type) and nm == "AdamW":
            cur_opt = v
            break
    if cur_opt is not None:
        ostep = cur_opt.step

        def step(self, *ar, **kw):
            STEPS.append((CUR["flops"], CUR["calls"], list(CUR["ns"])))
            CUR["flops"] = 0.0
            CUR["calls"] = 0
            CUR["ns"] = []
            return ostep(self, *ar, **kw)

        cur_opt.step = step

    sys.argv = ["v3_pretrain",
                "--data", r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512",
                "--vocab", "96000", "--seq", str(a.seq), "--d", "1024",
                "--layers", str(a.layers), "--E", "4", "--n_q", "16",
                "--d_ff", str(a.d_ff), "--top_k", "2", "--csa", "4",
                "--steps", str(a.steps), "--log-every", "1", "--no-final-save"]
    try:
        P.main()
    except SystemExit:
        pass

    print("\n=== 每步 GEMM 总量 ===")
    print(" step | GEMM 调用 | 总 GFLOP | n 的最小/最大值")
    for i, (f, c, ns) in enumerate(STEPS):
        arr = np.asarray(ns) if ns else np.asarray([0])
        print(" %4d | %9d | %8.1f | %d / %d" % (i, c, f / 1e9, arr.min(), arr.max()))


if __name__ == "__main__":
    main()
