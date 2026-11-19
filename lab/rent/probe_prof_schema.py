# -*- coding: utf-8 -*-
"""诊断 torch.profiler 在本机（V100 / torch 2.6.0+cu124）到底吐了什么：

  * events() 里 device_type 的分布
  * 有没有真正的 CUDA kernel 行（name / count / self_device_time / self_cpu_time）
  * key_averages() 的行长什么样

跑法：C:\\Python311\\python.exe -X utf8 D:\\orv_bench\\probe_prof_schema.py
"""
import collections
import os
import sys

WORK = r"D:\orv_bench"
V3DIR = r"D:/TGAI-V3-1.5B/03_src"
os.chdir(WORK)
sys.path.insert(0, V3DIR)
sys.path.insert(0, WORK)

import torch                                                    # noqa: E402
from torch.profiler import profile, ProfilerActivity            # noqa: E402
from torch.autograd import DeviceType                           # noqa: E402

import model_v3 as M                                            # noqa: E402

cfg = M.TGAIV3Config(vocab_size=96000, d_model=1024, n_layers=12, n_heads=16,
                     n_kv_heads=2, d_ff=3328, n_experts=4, n_activated=2,
                     n_shared_experts=1, max_seq_len=2048, moe_balance="sign")
model = M.TGAIV3Model(cfg).cuda()
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = torch.amp.GradScaler("cuda", enabled=True)
ids = torch.randint(0, 96000, (1, 2048), device="cuda")


def step():
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
        logits, loss, _ = model(ids, targets=ids)
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()


step()
torch.cuda.synchronize()

import time                                                    # noqa: E402
_t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    step()
    torch.cuda.synchronize()
step_wall_ms = (time.perf_counter() - _t0) * 1000.0

evs = list(prof.events())
print("events:", len(evs))
print("device_type 分布:", collections.Counter(str(e.device_type) for e in evs))
print("DeviceType.CUDA =", int(DeviceType.CUDA))

cand = [e for e in evs if float(getattr(e, "device_time_total", 0) or 0) > 0]
print("\n有 device_time 的事件数:", len(cand))
print("其中 device_type 分布:", collections.Counter(str(e.device_type) for e in cand))
cand.sort(key=lambda e: -float(e.device_time_total))
print("\n%-70s %-14s %8s %10s %10s" % ("key", "devtype", "count", "dev_us", "cpu_us"))
for e in cand[:15]:
    print("%-70s %-14s %8d %10.1f %10.1f"
          % (e.key[:70], str(e.device_type), 1, float(e.device_time_total),
             float(getattr(e, "self_cpu_time_total", 0) or 0)))

ka = prof.key_averages()
print("\nkey_averages 行数:", len(ka))
rows = sorted(ka, key=lambda e: -float(e.self_device_time_total or 0))
print("%-60s %-12s %8s %10s %10s" % ("key", "devtype", "count", "self_dev_us", "self_cpu_us"))
for e in rows[:15]:
    print("%-60s %-12s %8d %10.1f %10.1f"
          % (e.key[:60], str(getattr(e, "device_type", "?")), e.count,
             float(e.self_device_time_total or 0),
             float(e.self_cpu_time_total or 0)))

# ── 关键：torch 会把同一段 kernel 时间同时记到父 range 和子 op 上（实测
#   Optimizer.step 115 ms + 它的 _foreach_* 子 op 各 10-16 ms ⇒ 重复计 2.5×）。
#   只取**叶子**事件（无子节点）的 self_device_time 才能得到不重复的 GPU 时间。
leaf = [e for e in evs if len(getattr(e, "cpu_children", ())) == 0 and
        float(getattr(e, "self_device_time_total", 0) or 0) > 0]
tot_leaf = sum(float(e.self_device_time_total) for e in leaf)
tot_all = sum(float(getattr(e, "self_device_time_total", 0) or 0) for e in evs)
print("\n叶子事件数 %d ；叶子 self_dev 合计 %.1f ms ；全部事件 self_dev 合计 %.1f ms"
      % (len(leaf), tot_leaf / 1000.0, tot_all / 1000.0))
print("本步墙钟（含 profiler 开销）约 %.1f ms" % (step_wall_ms))
print("叶子 GPU / 墙钟 = %.3f" % (tot_leaf / 1000.0 / step_wall_ms))

agg = collections.Counter()
cnt = collections.Counter()
for e in leaf:
    agg[e.key] += float(e.self_device_time_total)
    cnt[e.key] += 1
print("\n叶子事件 top 25（不重复计）:")
print("%-62s %8s %10s %8s" % ("key", "count", "ms", "%leaf"))
for k, v in agg.most_common(25):
    print("%-62s %8d %10.2f %7.1f%%" % (k[:62], cnt[k], v / 1000.0,
                                        100.0 * v / tot_leaf))
