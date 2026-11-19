# -*- coding: utf-8 -*-
"""PyTorch 侧 kernel 归因（torch.profiler，CUDA activities）+ 事件分相计时。

口径与 ORV 侧严格对齐：
  V3 MoE d=1024 / d_ff=3328 / E=4(+1 shared) / top_k=2 / n_q=16 / n_kv=2 /
  csa=4（attn_global_every=4 ⇒ 3 global + 9 CSA）/ vocab=96000 / seq=2048 /
  batch=1 / 12 层 / AMP=autocast fp16 + GradScaler / ckpt 关 / AdamW lr 3e-4。

产出（stdout 里的 JSON 行 `PYT_PROF {...}`）：
  * wall：预热后 N 步的步时（单次 sync 与「4 次内部 sync」两种量法都给）
  * phase：forward / backward / optimizer 三段的事件毫秒（每步平均）
  * kernels：全部 kernel 的 name / count / self CUDA µs（每步平均）
  * cuda_per_step_ms / wall_ratio：CUDA 忙碌时间 ÷ 墙钟步时

跑法（远端）：
  C:\\Python311\\python.exe -X utf8 D:\\orv_bench\\prof_pyt.py --seq 2048 --steps 20
"""
import argparse
import json
import os
import statistics
import sys
import time

WORK = r"D:\orv_bench"
V3DIR = r"D:/TGAI-V3-1.5B/03_src"
os.chdir(WORK)
sys.path.insert(0, V3DIR)
sys.path.insert(0, WORK)

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402

import model_v3 as M                                           # noqa: E402


def build(seq, vocab, d, layers, n_heads, d_ff, E, top_k, ckpt, amp):
    cfg = M.TGAIV3Config(vocab_size=vocab, d_model=d, n_layers=layers,
                         n_heads=n_heads, n_kv_heads=2, d_ff=d_ff,
                         n_experts=E, n_activated=top_k, n_shared_experts=1,
                         max_seq_len=seq, moe_balance="sign")
    model = M.TGAIV3Model(cfg).cuda()
    model.enable_gradient_checkpointing(bool(ckpt))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp))
    return model, opt, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--prof-steps", type=int, default=20)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--d_ff", type=int, default=3328)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=96000)
    ap.add_argument("--ckpt", type=int, default=0)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--out", default=r"D:\orv_bench\prof_pyt_2048.json")
    a = ap.parse_args()

    model, opt, scaler = build(a.seq, a.vocab, a.d, a.layers, a.n_heads,
                               a.d_ff, a.E, a.top_k, a.ckpt, a.amp)
    n_par = sum(p.numel() for p in model.parameters())
    print("[pyt] params=%.3fM  seq=%d  amp=%d  ckpt=%d  vocab=%d"
          % (n_par / 1e6, a.seq, a.amp, a.ckpt, a.vocab), flush=True)

    rng = np.random.default_rng(0)
    total = a.warmup + a.steps + a.prof_steps
    ids_all = rng.integers(0, a.vocab, (total, a.seq)).astype(np.int64)
    ids_gpu = [torch.from_numpy(ids_all[i]).cuda()[None, :] for i in range(total)]

    def step(i, ev=None):
        """一步。ev 给定时：用 CUDA event 分段（不 sync），步末统一读。"""
        ids = ids_gpu[i]
        if ev is not None:
            ev[0].record()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=bool(a.amp)):
            logits, loss, _ = model(ids, targets=ids)
        if ev is not None:
            ev[1].record()
        opt.zero_grad()
        if ev is not None:
            ev[2].record()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if ev is not None:
            ev[3].record()
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if ev is not None:
            ev[4].record()
        return float(loss.detach())

    # ── 1) 预热 ────────────────────────────────────────────────────────────
    for i in range(a.warmup):
        step(i)
    torch.cuda.synchronize()

    # ── 2) 墙钟（单次 sync，每步一个 CUDA event 分段但不读，步末 sync）──────
    #   注意：torch 的 `Event.elapsed_time` 返回的就是**毫秒**（不是秒）。
    evs = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    walls, seg_f, seg_b, seg_o = [], [], [], []
    losses = []
    for k in range(a.steps):
        i = a.warmup + k
        t0 = time.perf_counter()
        losses.append(step(i, evs))
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        seg_f.append(evs[0].elapsed_time(evs[1]))
        seg_b.append(evs[2].elapsed_time(evs[3]))
        seg_o.append(evs[3].elapsed_time(evs[4]))
    med = statistics.median(walls)
    print("[pyt] wall single-sync: median=%.4f min=%.4f max=%.4f spread=%.1f%%"
          % (med, min(walls), max(walls),
             100.0 * (max(walls) - min(walls)) / med), flush=True)
    print("[pyt] phase median ms: fwd=%.1f bwd=%.1f opt=%.1f (sum=%.1f)"
          % (statistics.median(seg_f), statistics.median(seg_b),
             statistics.median(seg_o),
             statistics.median(seg_f) + statistics.median(seg_b)
             + statistics.median(seg_o)), flush=True)

    # ── 3) torch.profiler（CUDA + CPU activities；CUDA-only 在 2.6 会 assert）──
    #   ★ 只取**叶子** CPU 事件（无 cpu_children）上的 self_device_time：
    #     torch 的 range 归因会把同一段 kernel 时间同时记到父 range 和子 op 上
    #     （实测「全事件求和」665.9 ms vs 叶子 366.0 ms，重复计 1.8×），
    #     叶子求和才是不重复的 GPU 时间。
    from torch.profiler import profile, ProfilerActivity
    from torch.autograd import DeviceType
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for k in range(a.prof_steps):
            step(a.warmup + a.steps + k)
        torch.cuda.synchronize()
    prof_wall = time.perf_counter() - t0

    acc = {}
    n_cpu_ops = 0
    tot_all_us = 0.0
    for ev in prof.events():
        dev = float(getattr(ev, "self_device_time_total", 0) or 0.0)     # µs
        if ev.device_type == DeviceType.CPU:
            n_cpu_ops += 1
        tot_all_us += dev
        if len(getattr(ev, "cpu_children", ())) != 0 or dev <= 0:
            continue
        r = acc.get(ev.key)
        if r is None:
            acc[ev.key] = r = [0, 0.0, 0.0]                          # n, dev_us, cpu_us
        r[0] += 1
        r[1] += dev
        r[2] += float(getattr(ev, "self_cpu_time_total", 0) or 0.0)
    rows = [{"name": k, "count": v[0], "us_total": v[1],
             "us_per_step": v[1] / float(a.prof_steps),
             "cpu_us_per_step": v[2] / float(a.prof_steps)}
            for k, v in acc.items()]
    rows.sort(key=lambda r: -r["us_per_step"])
    cuda_per_step_ms = sum(r["us_per_step"] for r in rows) / 1000.0
    n_kern_per_step = sum(r["count"] for r in rows) / float(a.prof_steps)
    cpu_ops_per_step = n_cpu_ops / float(a.prof_steps)

    print("[pyt] profiler: %d steps, wall=%.3fs (%.1f ms/step, 含 profiler 开销)"
          % (a.prof_steps, prof_wall, 1000.0 * prof_wall / a.prof_steps), flush=True)
    print("[pyt] CUDA busy %.1f ms/step  vs wall %.1f ms/step  => ratio %.3f"
          % (cuda_per_step_ms, med * 1000.0, cuda_per_step_ms / (med * 1000.0)),
          flush=True)
    print("[pyt] kernel launches %.1f /step, distinct %d, CPU aten 事件 %.0f /step"
          % (n_kern_per_step, len(rows), cpu_ops_per_step), flush=True)

    res = {"impl": "pytorch", "cfg": {"seq": a.seq, "d": a.d, "layers": a.layers,
                                      "d_ff": a.d_ff, "E": a.E, "top_k": a.top_k,
                                      "vocab": a.vocab, "amp": a.amp, "ckpt": a.ckpt},
           "n_params": int(n_par), "steps": a.steps, "warmup": a.warmup,
           "prof_steps": a.prof_steps,
           "wall_median_s": med, "wall_min_s": min(walls), "wall_max_s": max(walls),
           "wall_all": walls, "losses": losses,
           "phase_ms": {"forward": statistics.median(seg_f),
                        "backward": statistics.median(seg_b),
                        "optimizer": statistics.median(seg_o)},
           "cuda_ms_per_step": cuda_per_step_ms,
           "cuda_over_wall": cuda_per_step_ms / (med * 1000.0),
           "cpu_ops_per_step": cpu_ops_per_step,
           "sum_all_events_us": tot_all_us / float(a.prof_steps),
           "prof_wall_s": prof_wall,
           "kernels_per_step": n_kern_per_step,
           "cpu_ops_per_step": cpu_ops_per_step,
           "kernels": rows}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("PYT_PROF " + json.dumps({k: v for k, v in res.items() if k != "kernels"},
                                   ensure_ascii=False), flush=True)
    print("[pyt] wrote %s" % a.out, flush=True)
    print("\ntop 30 kernels:")
    for r in rows[:30]:
        print("  %10.3f ms/step %7.1f%%  %6.1f 次/步  %s"
              % (r["us_per_step"] / 1000.0, 0.0, r["count"] / a.prof_steps,
                 r["name"][:96]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
