# -*- coding: utf-8 -*-
"""PyTorch 侧链路拆解：同配置一步训练，统计内核调用与时间分布。

与 orv_profile.py 对照用（同 d / layers / seq / csa / vocab）。

在 V100 上跑：
    C:\\Python311\\python.exe pyt_profile.py --layers 4
"""
import argparse
import sys

sys.path.insert(0, r"D:\TGAI-V3-1.5B\03_src")   # model_v3.py 在真项目里

import torch

import model_v3 as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n_q", type=int, default=16)
    ap.add_argument("--d_ff", type=int, default=2816)
    ap.add_argument("--vocab", type=int, default=96000)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--csa", type=int, default=4)
    a = ap.parse_args()

    cfg = M.TGAIV3Config(
        vocab_size=a.vocab, d_model=a.d, n_layers=a.layers, n_heads=a.n_q,
        n_kv_heads=0, d_ff=a.d_ff, n_experts=a.E, n_activated=2,
        n_shared_experts=1, max_seq_len=a.seq,
        csa_compress=4, csa_conv_kernel=3, csa_local_window=0, csa_latent=0,
        attn_global_every=a.csa, rope_theta=500000.0)
    torch.manual_seed(0)
    model = M.TGAIV3Model(cfg).cuda().train()
    model.enable_gradient_checkpointing(True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ids = torch.randint(0, a.vocab, (1, a.seq), device="cuda")
    tgt = torch.roll(ids, -1, dims=1)

    def one_step():
        out = model(ids, targets=ids)
        loss = out[1] if isinstance(out, (tuple, list)) else out
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

    one_step()          # 预热

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        one_step()

    evts = prof.key_averages()
    # 只统计 GPU 内核
    rows = []
    for e in evts:
        t = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if t > 0:
            rows.append((e.key, int(e.count), t / 1000.0))
    rows.sort(key=lambda r: -r[2])
    total_cnt = sum(r[1] for r in rows)
    total_ms = sum(r[2] for r in rows)
    print("=== PyTorch 一步（layers=%d d=%d seq=%d csa=%d）==="
          % (a.layers, a.d, a.seq, a.csa))
    print("GPU 内核总次数: %d   总 GPU 时间: %.1f ms" % (total_cnt, total_ms))
    print("\n按 GPU 时间排序（前 20）:")
    for k, c, ms in rows[:20]:
        print("  %-46s cnt=%5d  %8.2f ms  (%.1f%%)"
              % (k[:46], c, ms, ms * 100.0 / max(total_ms, 1e-9)))


if __name__ == "__main__":
    main()
