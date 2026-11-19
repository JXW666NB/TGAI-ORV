# -*- coding: utf-8 -*-
"""PyT 侧「分相 kernel 时间」：forward / backward / optimizer 各开独立 profiler 窗口。

为什么要单独一个脚本：一次 profiler 包住整步只能拿到全步合计；要回答「前向里
GPU 到底忙了多久、空转多久」必须分相。做法：
  * forward 相：窗口内只跑前向（不 backward），窗口结束立刻释放图；
  * backward 相：前向在窗口**外**建图，窗口内只跑 backward；
  * optimizer 相：前向+反向都在窗口外，窗口内只跑 opt.step()。
每相跑 N 步，取**叶子事件**的 self_device_time 求和（避免 torch 的 range 归因重复计）。
窗口外的那部分 kernel 不会被 CUPTI 记录，所以每相的数字是干净的单相 kernel 时间。

跑法：C:\\Python311\\python.exe -X utf8 D:\\orv_bench\\prof_pyt_phases.py --seq 2048 --steps 3
"""
import argparse
import json
import os
import statistics
import sys

WORK = r"D:\orv_bench"
V3DIR = r"D:/TGAI-V3-1.5B/03_src"
os.chdir(WORK)
sys.path.insert(0, V3DIR)
sys.path.insert(0, WORK)

import torch                                                   # noqa: E402
from torch.profiler import profile, ProfilerActivity           # noqa: E402
from torch.autograd import DeviceType                          # noqa: E402

import model_v3 as M                                           # noqa: E402


def leaf_dev_us(prof):
    acc = {}
    for ev in prof.events():
        if ev.device_type != DeviceType.CPU:
            continue
        if len(getattr(ev, "cpu_children", ())) != 0:
            continue
        d = float(getattr(ev, "self_device_time_total", 0) or 0.0)
        if d <= 0:
            continue
        acc[ev.key] = acc.get(ev.key, 0.0) + d
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--d_ff", type=int, default=3328)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=96000)
    ap.add_argument("--out", default=r"D:\orv_bench\prof_pyt_phases.json")
    a = ap.parse_args()

    cfg = M.TGAIV3Config(vocab_size=a.vocab, d_model=a.d, n_layers=a.layers,
                         n_heads=a.n_heads, n_kv_heads=2, d_ff=a.d_ff,
                         n_experts=a.E, n_activated=a.top_k, n_shared_experts=1,
                         max_seq_len=a.seq, moe_balance="sign")
    model = M.TGAIV3Model(cfg).cuda()
    model.enable_gradient_checkpointing(False)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    ids = [torch.randint(0, a.vocab, (1, a.seq), device="cuda")
           for _ in range(a.steps + 2)]

    def fwd(i):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            return model(ids[i], targets=ids[i])

    # 预热
    for i in range(2):
        logits, loss, _ = fwd(i)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()

    res = {}
    # ① forward
    acc = {}
    for i in range(a.steps):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pf:
            logits, loss, _ = fwd(2 + i)
            torch.cuda.synchronize()
        for k, v in leaf_dev_us(pf).items():
            acc[k] = acc.get(k, 0.0) + v
        del logits, loss
        opt.zero_grad(set_to_none=True)
    res["forward"] = {k: v / a.steps for k, v in acc.items()}

    # ② backward
    acc = {}
    for i in range(a.steps):
        opt.zero_grad(set_to_none=True)
        logits, loss, _ = fwd(2 + i)                 # 窗口外建图
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pf:
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
        for k, v in leaf_dev_us(pf).items():
            acc[k] = acc.get(k, 0.0) + v
        del logits, loss
    res["backward"] = {k: v / a.steps for k, v in acc.items()}

    # ③ optimizer（含 GradScaler.step/update，与训练循环一致）
    acc = {}
    for i in range(a.steps):
        opt.zero_grad(set_to_none=True)
        logits, loss, _ = fwd(2 + i)
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pf:
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
        for k, v in leaf_dev_us(pf).items():
            acc[k] = acc.get(k, 0.0) + v
        del logits, loss
    res["optimizer"] = {k: v / a.steps for k, v in acc.items()}

    out = {"steps": a.steps, "seq": a.seq,
           "phase_ms": {p: sum(v.values()) / 1000.0 for p, v in res.items()},
           "phase_kernels": {p: {k: v / 1000.0 for k, v in d.items()}
                             for p, d in res.items()}}
    for p in ("forward", "backward", "optimizer"):
        print("%-10s kernel 合计 %8.2f ms  (%d 个不同 op)"
              % (p, out["phase_ms"][p], len(res[p])))
        top = sorted(res[p].items(), key=lambda kv: -kv[1])[:8]
        for k, v in top:
            print("      %8.2f ms  %s" % (v / 1000.0, k))
    print("PHASES " + json.dumps(out["phase_ms"], ensure_ascii=False))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[wrote] %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
