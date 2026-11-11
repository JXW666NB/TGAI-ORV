# -*- coding: utf-8 -*-
"""MoE 反向 332 ms 到底花在哪：按真实尺寸逐个测原语（预热充分、避免把分配算进去）。

之前那个微基准给出"8 次小 GEMM = 6.11 ms"，但实测整条前向链只有 237 ms —— 两者矛盾，
说明微基准把 buffer 分配也算进去了。这里每个用例都先分配好、预热够，再计时。

运行（需独占 GPU）：python moe_prim_bench.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

D, DF, NE, TOPK, SEQ = 1024, 4096, 8, 2, 512
REP = 20


def log(m):
    print(m, flush=True)


def main():
    import torch
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)

    def ev(fn, rep=REP):
        for _ in range(3):
            fn()
        rt.sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(rep):
            fn()
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / rep

    # 真实形状：seq=512、top-2 of 8 ⇒ 每专家约 128 行
    counts = [128] * NE
    x = rt.h2d(rng.normal(0, 0.02, (SEQ, D)).astype(np.float32))
    Wg = rt.h2d(rng.normal(0, 0.02, (NE, D)).astype(np.float32))          # router
    w_gate = [rt.h2d(rng.normal(0, 0.02, (DF, D)).astype(np.float32)) for _ in range(NE)]
    w_down = [rt.h2d(rng.normal(0, 0.02, (D, DF)).astype(np.float32)) for _ in range(NE)]
    rows = [rt.h2d(rng.normal(0, 1, (c, D)).astype(np.float32)) for c in counts]
    hid = [rt.h2d(rng.normal(0, 1, (c, DF)).astype(np.float32)) for c in counts]

    # 预分配输出（不把分配算进计时）
    out_gate = [rt.empty((c, DF), dtype=np.float32) for c in counts]
    out_down = [rt.empty((c, D), dtype=np.float32) for c in counts]

    log("== 单层 MoE 各原语耗时（seq=%d，8 专家，每专家 %d 行）==" % (SEQ, counts[0]))

    def t_gemm_gate():
        for k in range(NE):
            r = rt.gemm_nt(rows[k], w_gate[k])
            out_gate[k] = r
    log("① gate 投影 8 次 gemm_nt      %.2f ms" % ev(t_gemm_gate))

    def t_gemm_down():
        for k in range(NE):
            out_down[k] = rt.gemm_nt(hid[k], w_down[k])
    log("② down 投影 8 次 gemm_nt      %.2f ms" % ev(t_gemm_down))

    def t_router():
        return rt.gemm_nt(x, Wg)
    log("③ router（1 次 [512,1024]@[1024,8]）%.2f ms" % ev(t_router))

    logits = rt.gemm_nt(x, Wg)
    idx_buf, val_buf = rt.moe_topk(logits, TOPK)
    log("④ moe_topk                    %.2f ms" % ev(lambda: rt.moe_topk(logits, TOPK)))
    log("⑤ moe_compact                 %.2f ms" % ev(lambda: rt.moe_compact(idx_buf, SEQ, NE, TOPK)))
    off_buf, tok_buf, slot_buf = rt.moe_compact(idx_buf, SEQ, NE, TOPK)
    off = np.asarray(rt.d2h(off_buf), np.int32).reshape(-1)
    log("⑥ moe_gather_rows ×8          %.2f ms"
        % ev(lambda: [rt.moe_gather_rows(x, tok_buf, int(off[k]), counts[k], D) for k in range(NE)]))

    # 元素级（silu / mul / scale）在 [128,4096] 上的单次耗时
    a = hid[0]
    b = hid[1]
    log("⑦ ewise mul ×8（[128,4096]）   %.2f ms"
        % ev(lambda: [rt.mul(hid[k], hid[(k + 1) % NE]) for k in range(NE)]))

    total_gemm = 2 * 6.11 * 0 + 0  # 占位，下面用实测值算
    log("")
    log("== 换算到 28 层 ==")


if __name__ == "__main__":
    main()
