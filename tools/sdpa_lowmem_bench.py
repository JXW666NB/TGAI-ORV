# -*- coding: utf-8 -*-
"""低显存 SDPA（不物化 P）vs 物化版：一致性 + 时间 + 显存（T=512，16 头，dh=64）。

运行（需独占 GPU）：python sdpa_lowmem_bench.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

T = 512
H = 16
DH = 64


def rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


def main():
    import torch
    import tgai_go.train as Tn
    Tn.set_default_device("cuda")
    rt = Tn._cuda()
    if not hasattr(Tn.functional, "sdpa_lowmem_ready"):
        log("没有 sdpa_lowmem"); return 2
    log("lowmem 可用: %s" % Tn.functional.sdpa_lowmem_ready())

    rng = np.random.default_rng(0)
    qn = rng.normal(0, 0.3, (1, H, T, DH)).astype(np.float32)
    kn = rng.normal(0, 0.3, (1, H, T, DH)).astype(np.float32)
    vn = rng.normal(0, 0.3, (1, H, T, DH)).astype(np.float32)
    gw = rng.normal(0, 1, (1, H, T, DH)).astype(np.float32)

    def run(lowmem):
        q = Tn.tensor(qn, requires_grad=True)
        k = Tn.tensor(kn, requires_grad=True)
        v = Tn.tensor(vn, requires_grad=True)
        y = Tn.functional.scaled_dot_product_attention(q, k, v, causal=True, lowmem=lowmem)
        yn = np.asarray(y.to_numpy(), np.float32)
        (y * Tn.tensor(gw)).sum().backward()
        return yn, (np.asarray(q.grad.to_numpy(), np.float32),
                    np.asarray(k.grad.to_numpy(), np.float32),
                    np.asarray(v.grad.to_numpy(), np.float32))

    y_mat, g_mat = run(False)
    rt.sync()
    y_low, g_low = run(True)
    rt.sync()
    log("前向相对差 %.2e" % rel(y_low, y_mat))
    log("dq %.2e  dk %.2e  dv %.2e"
        % (rel(g_low[0], g_mat[0]), rel(g_low[1], g_mat[1]), rel(g_low[2], g_mat[2])))

    def timeit(lowmem, rep=3):
        run(lowmem)
        torch.cuda.synchronize()
        ts = []
        for _ in range(rep):
            torch.cuda.synchronize(); t0 = time.time()
            run(lowmem)
            torch.cuda.synchronize(); ts.append(time.time() - t0)
        return min(ts) * 1000

    t_mat = timeit(False)
    t_low = timeit(True)
    log("时间：物化 %.1f ms → 低显存 %.1f ms（%.2fx，>1 表示低显存更慢）"
        % (t_mat, t_low, t_low / max(t_mat, 1e-9)))
    log("显存（live）：物化 %.2f GiB / 低显存 %.2f GiB" %
        (float(rt.live_bytes()) / (1 << 30) if hasattr(rt, "live_bytes") else -1, -1))


if __name__ == "__main__":
    main()
