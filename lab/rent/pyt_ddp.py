# -*- coding: utf-8 -*-
"""PyTorch 侧的双卡数据并行基线（DDP），用于与 ORV 的 DP 做同条件对比。

单进程启动即可：内部用 mp.spawn 起 world 个进程，nccl 通信。
每个 rank 用**不同**的随机数据（与 ORV 的分片取数对应）。

用法：
    python pyt_ddp.py --layers 12 --seq 512 --world 2
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def worker(rank, world, a):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(a.port)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    import model_v3 as M
    cfg = M.TGAIV3Config(vocab_size=a.vocab, d_model=a.d, n_layers=a.layers,
                         n_heads=a.n_heads, n_kv_heads=2, d_ff=a.d_ff,
                         n_experts=a.E, n_activated=a.top_k,
                         n_shared_experts=1, max_seq_len=a.seq,
                         moe_balance="sign")
    model = M.TGAIV3Model(cfg).to(dev)
    model.enable_gradient_checkpointing(True)
    model.train()
    ddp = DDP(model, device_ids=[rank], find_unused_parameters=True)   # MoE 路由是数据相关的，必然有专家本轮被跳过
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-4)
    n_par = sum(p.numel() for p in model.parameters())

    torch.cuda.reset_peak_memory_stats(dev)
    rng = np.random.default_rng(1000 + rank)      # ★ 每个 rank 不同数据
    ids = torch.from_numpy(rng.integers(0, a.vocab, (a.steps + 1, a.seq))
                           .astype(np.int64)).to(dev)

    times = []
    for i in range(a.steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = ddp(ids[i:i + 1], targets=ids[i:i + 1])
        loss = out[1]                      # (logits, loss)，与参考脚本一致
        loss.backward()
        opt.step()
        torch.cuda.synchronize(dev)
        if i >= 1:
            times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2] if times else float("nan")
    peak = torch.cuda.max_memory_allocated(dev) / (1 << 30)

    if rank == 0:
        print("RESULT " + json.dumps({
            "impl": "pytorch-ddp", "world": world, "layers": a.layers,
            "seq": a.seq, "vocab": a.vocab, "params": n_par,
            "step_s_steady": med, "tok_s_per_rank": a.seq / med,
            "tok_s_aggregate": a.seq * world / med,
            "peak_gib_per_card": peak,
        }))
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--d_ff", type=int, default=3328)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=96000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--port", type=int, default=29531)
    a = ap.parse_args()
    mp.spawn(worker, args=(a.world, a), nprocs=a.world, join=True)


if __name__ == "__main__":
    main()
