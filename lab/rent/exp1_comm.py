import os
import time

import torch
import torch.distributed as dist

"""
Experiment 1: allreduce cost for one step's gradients (848M fp32 = 3.4 GB).

A) NCCL all_reduce on the GPU (what PyTorch DDP uses)
B) host-staged: pinned D2H -> host sum -> H2D (what tgai_go.train.dist does)
   measured single-process (both halves on one rank, same bytes touched)

Reported for 848.6M params, which is the 12-layer V3 bench config.
"""

N_PARAMS = 848_645_257
BYTES = N_PARAMS * 4                      # fp32 gradient bytes = 3.16 GiB
SIZES_MB = [400, 3400]                    # medium probe + the real thing


def barrier():
    if dist.is_initialized():
        dist.barrier()
        torch.cuda.synchronize()


def bench_nccl(n_elems, iters=5):
    x = torch.ones(n_elems, dtype=torch.float32, device="cuda")
    for _ in range(2):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    barrier()
    dt = (time.perf_counter() - t0) / iters
    return dt


def bench_host_staged(n_elems, iters=5):
    """3.16 GiB fp32 grad: D2H pinned -> host reduce -> H2D, exactly the dist.py path."""
    g = torch.ones(n_elems, dtype=torch.float32, device="cuda")
    h = torch.empty(n_elems, dtype=torch.float32, pin_memory=True)
    acc = torch.zeros(n_elems, dtype=torch.float32, pin_memory=True)
    for _ in range(2):
        h.copy_(g, non_blocking=False)
        acc.add_(h)
        g.copy_(acc)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        h.copy_(g, non_blocking=False)     # D2H
        acc.add_(h)                        # host reduce
        g.copy_(acc)                       # H2D
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    rank = int(os.environ.get("RANK", 0))
    local = rank % max(1, torch.cuda.device_count())
    torch.cuda.set_device(local)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local))

    if rank == 0:
        print("world=%s  gpu=%s" % (os.environ.get("WORLD_SIZE", 1),
                                    torch.cuda.get_device_name(local)))
    for mb in SIZES_MB:
        n = mb * 1024 * 1024 // 4
        if dist.is_initialized():
            t = bench_nccl(n)
            if rank == 0:
                print("  NCCL allreduce  %5d MB  %7.2f ms  (%.1f GB/s busbw)"
                      % (mb, t * 1e3, 2 * (n * 4) * (dist.get_world_size() - 1)
                         / dist.get_world_size() / t / 1e9))
        t = bench_host_staged(n)
        if rank == 0:
            print("  host-staged     %5d MB  %7.2f ms  (%.1f GB/s 单向)"
                  % (mb, t * 1e3, (n * 4) / t / 1e9))
    if rank == 0:
        print("  ---- 一步的梯度是 %.2f GiB ----" % (BYTES / 2**30))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
