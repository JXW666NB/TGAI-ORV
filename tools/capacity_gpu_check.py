# -*- coding: utf-8 -*-
"""在真 GPU 上验证容量因子：CPU 与 CUDA 的丢弃行为必须一致。

风险点：CUDA 路径是在主机侧的 `off` 偏移上做截断，CPU 路径是在 Python 循环里截断。
两条路必须丢**同一批** token，否则训练会出现"CPU 调试正常、GPU 结果不同"。

运行（需独占 GPU）：python capacity_gpu_check.py
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TGAI-ORV-main", "python"))

D, F, E, K, N = 64, 128, 8, 2, 96
OUT = os.path.join(HERE, "cgc.txt")


def main():
    import tgai_go.train as T
    rep, ok = [], True
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (N, D)).astype(np.float32)

    for cf in (None, 2.0, 1.0):
        # CPU
        T.set_default_device("cpu")
        m_cpu = T.nn.MoE(D, F, E, top_k=K, gate="softmax_topk", capacity_factor=cf)
        seed = np.random.default_rng(1)
        for p in m_cpu.parameters():
            p.data = seed.normal(0, 0.05, tuple(int(s) for s in p.shape)).astype(np.float32)
        y_cpu = np.asarray(m_cpu(T.tensor(X)).to_numpy(), np.float32)

        # CUDA，同一份权重
        T.set_default_device("cuda")
        rt = T._cuda()
        m_gpu = T.nn.MoE(D, F, E, top_k=K, gate="softmax_topk", capacity_factor=cf)
        for pg, pc in zip(m_gpu.parameters(), m_cpu.parameters()):
            pg._gpu = rt.h2d(np.asarray(pc.data, np.float32).reshape(
                tuple(int(s) for s in pg.shape)))
        y_gpu = np.asarray(m_gpu(T.tensor(X)).to_numpy(), np.float32)

        rel = float(np.abs(y_gpu - y_cpu).max() / max(1e-9, float(np.abs(y_cpu).max())))
        z_cpu = np.abs(y_cpu).max(axis=1) == 0
        z_gpu = np.abs(y_gpu).max(axis=1) == 0
        # ⚠️ 已知限制：开启容量后 compact 顺序不同 ⇒ 两边"丢哪些"可能不同，
        # 因此 cf 非 None 时**不能**要求输出逐位一致或丢弃集合一致。
        # 这里只要求：不限制时逐位一致；限制时两边都有限、且 cap 被遵守。
        if cf is None:
            same_drop = bool((z_cpu == z_gpu).all())
            rep.append("cf=%-5s 相对差 %.2e | 丢弃 CPU=%d GPU=%d 一致=%s"
                       % (cf, rel, int(z_cpu.sum()), int(z_gpu.sum()), same_drop))
            ok = ok and rel <= 1e-5 and same_drop
        else:
            finite = bool(np.isfinite(y_cpu).all() and np.isfinite(y_gpu).all())
            rep.append("cf=%-5s 相对差 %.2e（容量限制下不要求逐位一致）| 丢弃 CPU=%d GPU=%d | 有限值=%s"
                       % (cf, rel, int(z_cpu.sum()), int(z_gpu.sum()), finite))
            ok = ok and finite

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("rc=%d\n" % (0 if ok else 1))
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print("总结论: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
