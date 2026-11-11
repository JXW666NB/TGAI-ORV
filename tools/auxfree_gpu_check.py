# -*- coding: utf-8 -*-
"""在真 GPU 上验证 aux-loss-free：CUDA 路径与 CPU 一致，且 bias 不进权重。

运行（需独占 GPU）：python auxfree_gpu_check.py
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TGAI-ORV-main", "python"))

D, F, E, K, N = 64, 128, 8, 2, 128
OUT = os.path.join(HERE, "agc.txt")


def log(m):
    print(m, flush=True)


def main():
    import tgai_go.train as T
    ok = True
    rep = []
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (N, D)).astype(np.float32)

    # ---------- 1) CPU 参考（bias 固定成非零值，确保走的是 aux_free 分支） ----------
    T.set_default_device("cpu")
    m_cpu = T.nn.MoE(D, F, E, top_k=K, gate="softmax", balance="aux_free")
    seed = np.random.default_rng(1)
    for p in m_cpu.parameters():
        p.data = seed.normal(0, 0.05, tuple(int(s) for s in p.shape)).astype(np.float32)
    bias = np.array([0.3, -0.2, 0.1, 0.0, 0.25, -0.15, 0.05, 0.0], np.float32)
    m_cpu.expert_bias = bias.copy()
    ref = np.asarray(m_cpu(T.tensor(X)).to_numpy(), np.float32)

    # ---------- 2) CUDA 同一份权重与 bias ----------
    T.set_default_device("cuda")
    rt = T._cuda()
    m_gpu = T.nn.MoE(D, F, E, top_k=K, gate="softmax", balance="aux_free")
    for pg, pc in zip(m_gpu.parameters(), m_cpu.parameters()):
        pg._gpu = rt.h2d(np.asarray(pc.data, np.float32).reshape(
            tuple(int(s) for s in pg.shape)))
    m_gpu.expert_bias = bias.copy()
    m_gpu.training = False          # 关掉 bias 更新，只比前向
    m_cpu.training = False
    out = np.asarray(m_gpu(T.tensor(X)).to_numpy(), np.float32)
    rel = float(np.abs(out - ref).max() / max(1e-9, float(np.abs(ref).max())))
    rep.append("CPU vs CUDA（aux_free，固定 bias）相对差 = %.2e ⇒ %s"
               % (rel, "一致" if rel <= 1e-5 else "**不一致**"))
    ok = ok and rel <= 1e-5

    # ---------- 3) CUDA 上 bias 更新生效且方向正确 ----------
    m_gpu.training = True
    m_gpu.expert_bias = np.array([5.0, 0, 0, 0, 0, 0, 0, 0], np.float32)
    m_gpu(T.tensor(X))
    b = m_gpu.expert_bias
    down = b[0] < 5.0
    up = (b[1:] > 0).all()
    rep.append("CUDA 上 bias 更新：过载专家下降=%s，欠载专家上升=%s" % (down, up))
    ok = ok and down and up

    # ---------- 4) CUDA 上 bias 不进门控权重（bias 归零 == balance=none） ----------
    m_zero = T.nn.MoE(D, F, E, top_k=K, gate="softmax", balance="aux_free")
    m_none = T.nn.MoE(D, F, E, top_k=K, gate="softmax", balance="none")
    for a, bb in zip(m_zero.parameters(), m_none.parameters()):
        a._gpu = rt.h2d(np.asarray(seed.normal(0, 0.05, tuple(int(s) for s in a.shape)), np.float32))
    for a, bb in zip(m_zero.parameters(), m_none.parameters()):
        bb._gpu = rt.h2d(np.asarray(rt.d2h(a._gpu), np.float32))
    m_zero.training = False
    m_none.training = False
    m_zero.expert_bias = np.zeros(E, np.float32)
    yz = np.asarray(rt.d2h(m_zero(T.tensor(X))._gpu), np.float32)
    yn = np.asarray(rt.d2h(m_none(T.tensor(X))._gpu), np.float32)
    d = float(np.abs(yz - yn).max())
    rep.append("bias 归零后与 balance=none 的最大绝对差 = %.2e ⇒ %s"
               % (d, "完全等价" if d == 0.0 else "**不等价**"))
    ok = ok and (d == 0.0)

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("rc=%d\n" % (0 if ok else 1))
        f.write("\n".join(rep) + "\n")
    log("\n".join(rep))
    log("总结论: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
