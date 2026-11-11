# -*- coding: utf-8 -*-
"""在真 GPU 上验证共享专家：CUDA 路径与 CPU 路径数值一致，并测它的开销。

为什么必须测：ORV 的纪律是 CPU/CUDA 两条路径行为一致。共享专家加在 `_forward_flat`
这一层，理论上两条路自动同时生效 —— 但"理论上"不算数，要实测。

运行（需独占 GPU）：python shared_gpu_check.py
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TGAI-ORV-main", "python"))

D, F, E, K, SEQ = 256, 512, 8, 2, 512
OUT = os.path.join(HERE, "sgc.txt")


def log(m):
    print(m, flush=True)


def main():
    import torch
    import tgai_go.train as T

    ok = True
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (SEQ, D)).astype(np.float32)

    report = []

    # ---------- 1) CPU 参考 ----------
    T.set_default_device("cpu")
    m_cpu = T.nn.MoE(D, F, E, top_k=K, gate="softmax_topk", n_shared=1)
    seed = np.random.default_rng(1)
    for p in m_cpu.parameters():
        p.data = seed.normal(0, 0.02, tuple(int(s) for s in p.shape)).astype(np.float32)
    ref = np.asarray(m_cpu(T.tensor(X)).to_numpy(), np.float32)

    # ---------- 2) CUDA 同一份权重 ----------
    T.set_default_device("cuda")
    rt = T._cuda()
    m_gpu = T.nn.MoE(D, F, E, top_k=K, gate="softmax_topk", n_shared=1)
    sd = m_cpu.state_dict()
    try:
        m_gpu.load_state_dict(sd)
    except Exception as exc:
        report.append("load_state_dict 失败: %s" % str(exc)[:140])
        # 退回逐参数 h2d
        for pg, pc in zip(m_gpu.parameters(), m_cpu.parameters()):
            pg._gpu = rt.h2d(np.asarray(pc.data, np.float32).reshape(tuple(int(s) for s in pg.shape)))
    out = np.asarray(m_gpu(T.tensor(X)).to_numpy(), np.float32)
    rel = float(np.abs(out - ref).max() / max(1e-9, float(np.abs(ref).max())))
    report.append("CPU vs CUDA（n_shared=1）相对差 = %.2e ⇒ %s"
                  % (rel, "一致" if rel <= 1e-5 else "**不一致**"))
    ok = ok and rel <= 1e-5

    # ---------- 3) 共享专家的开销（n_shared=0 vs 1 vs 2） ----------
    def ev(fn, rep=10):
        for _ in range(3):
            fn()
        rt.sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(rep):
            fn()
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / rep

    for ns in (0, 1, 2):
        m = T.nn.MoE(D, F, E, top_k=K, gate="softmax_topk", n_shared=ns)
        for p in m.parameters():
            p._gpu = rt.h2d(np.asarray(seed.normal(0, 0.02, tuple(int(s) for s in p.shape)), np.float32))
        t = ev(lambda: m(T.tensor(X)))
        report.append("n_shared=%d 前向 %.2f ms" % (ns, t))

    # ---------- 4) 梯度能到共享专家（GPU） ----------
    m = T.nn.MoE(D, F, E, top_k=K, gate="softmax_topk", n_shared=1)
    for p in m.parameters():
        p._gpu = rt.h2d(np.asarray(seed.normal(0, 0.02, tuple(int(s) for s in p.shape)), np.float32))
    y = m(T.tensor(X))
    y.sum().backward()
    g = m.shared_experts[0].down.weight.grad
    has = g is not None
    nz = has and float(np.abs(np.asarray(rt.d2h(g._gpu), np.float32)).max()) > 0
    report.append("GPU 上共享专家拿到梯度 = %s" % (has and nz))
    ok = ok and has and nz

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("rc=%d\n" % (0 if ok else 1))
        f.write("\n".join(report) + "\n")
    log("\n".join(report))
    log("总结论: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
