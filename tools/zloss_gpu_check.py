# -*- coding: utf-8 -*-
"""在真 GPU 上验证 Z-loss：CPU 与 CUDA 数值一致、梯度能到 router。

运行（需独占 GPU）：python zloss_gpu_check.py
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TGAI-ORV-main", "python"))

D, F, E, K, N = 64, 128, 8, 2, 96
OUT = os.path.join(HERE, "zgc.txt")


def main():
    import tgai_go.train as T
    rep, ok = [], True
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (N, D)).astype(np.float32)
    coef = 1e-3

    # CPU
    T.set_default_device("cpu")
    m_cpu = T.nn.MoE(D, F, E, top_k=K, gate="softmax", z_loss_coef=coef)
    seed = np.random.default_rng(1)
    for p in m_cpu.parameters():
        p.data = seed.normal(0, 0.05, tuple(int(s) for s in p.shape)).astype(np.float32)
    m_cpu(T.tensor(X))
    z_cpu = float(np.asarray(m_cpu.z_loss.data))

    # CUDA 同一份权重
    T.set_default_device("cuda")
    rt = T._cuda()
    m_gpu = T.nn.MoE(D, F, E, top_k=K, gate="softmax", z_loss_coef=coef)
    for pg, pc in zip(m_gpu.parameters(), m_cpu.parameters()):
        pg._gpu = rt.h2d(np.asarray(pc.data, np.float32).reshape(
            tuple(int(s) for s in pg.shape)))
    m_gpu(T.tensor(X))
    z_gpu_raw = m_gpu.z_loss
    # z_loss 在 CUDA 上是设备上的标量，拉回主机比较
    if hasattr(z_gpu_raw, "_gpu") and z_gpu_raw._gpu is not None:
        z_gpu = float(np.asarray(rt.d2h(z_gpu_raw._gpu), np.float32).reshape(-1)[0])
    else:
        z_gpu = float(np.asarray(z_gpu_raw.data, np.float32))
    rel = abs(z_cpu - z_gpu) / max(1e-9, abs(z_cpu))
    rep.append("z_loss：CPU %.8f vs CUDA %.8f，相对差 %.2e ⇒ %s"
               % (z_cpu, z_gpu, rel, "一致" if rel <= 1e-5 else "**不一致**"))
    ok = ok and rel <= 1e-5

    # GPU 上梯度能到 router
    m_gpu.z_loss.backward()
    g = m_gpu.router.weight.grad
    got = g is not None
    if got:
        arr = np.asarray(rt.d2h(g._gpu), np.float32) if hasattr(g, "_gpu") and g._gpu is not None \
            else np.asarray(g, np.float32)
        got = float(np.abs(arr).max()) > 0
    rep.append("GPU 上 z_loss 的梯度到达 router = %s" % got)
    ok = ok and got

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("rc=%d\n" % (0 if ok else 1))
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print("总结论: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
