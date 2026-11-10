# -*- coding: utf-8 -*-
"""低显存 SDPA vs 物化版：用 CUDA event 计时、用设备空闲显存差量显存（T=512，16 头，dh=64）。

运行（需独占 GPU）：python sdpa_mem_bench.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

T, H, DH = 512, 16, 64


def main():
    import torch
    import tgai_go.train as Tn
    Tn.set_default_device("cuda")
    rt = Tn._cuda()
    Tn.functional.sdpa_lowmem_ready()

    rng = np.random.default_rng(0)
    qn = rng.normal(0, 0.3, (1, H, T, DH)).astype(np.float32)
    kn = rng.normal(0, 0.3, (1, H, T, DH)).astype(np.float32)
    vn = rng.normal(0, 0.3, (1, H, T, DH)).astype(np.float32)
    gw = rng.normal(0, 1, (1, H, T, DH)).astype(np.float32)

    def build():
        return (Tn.tensor(qn, requires_grad=True), Tn.tensor(kn, requires_grad=True),
                Tn.tensor(vn, requires_grad=True), Tn.tensor(gw))

    def fwdbwd(lowmem):
        q, k, v, g = build()
        y = Tn.functional.scaled_dot_product_attention(q, k, v, causal=True, lowmem=lowmem)
        (y * g).sum().backward()
        return q, k, v

    def timed(lowmem, rep=5):
        fwdbwd(lowmem); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(rep):
            fwdbwd(lowmem)
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / rep

    def peak_mem(lowmem):
        """用设备空闲显存的最小值估算峰值（ORV 不走 torch 分配器）。"""
        rt.empty_cache(); torch.cuda.synchronize()
        base = rt.free_device_bytes()
        hold = []
        q, k, v, g = build()
        y = Tn.functional.scaled_dot_product_attention(q, k, v, causal=True, lowmem=lowmem)
        (y * g).sum().backward()
        rt.sync()
        used = base - rt.free_device_bytes()
        del hold
        return used

    t_mat, t_low = timed(False), timed(True)
    m_mat, m_low = peak_mem(False), peak_mem(True)
    log("时间（fwd+bwd，CUDA event）：物化 %.2f ms → 低显存 %.2f ms（%.2fx）"
        % (t_mat, t_low, t_low / max(t_mat, 1e-9)))
    log("显存（净增，含输入/梯度）：物化 %.1f MiB → 低显存 %.1f MiB（省 %.1f MiB）"
        % (m_mat / 1024 ** 2, m_low / 1024 ** 2, (m_mat - m_low) / 1024 ** 2))


if __name__ == "__main__":
    main()
