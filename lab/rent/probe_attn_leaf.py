# -*- coding: utf-8 -*-
"""核对 torch profiler 的两件事（只跑 1 步，便宜）：

1. `_efficient_attention_*` 是「非叶子 range」——它们的 device 时间是否**已经**
   被子 aten op 分摊（即叶子求和是否漏算注意力）。
2. 全事件 self_dev 求和 vs 叶子求和的比例（重复计倍数）。

跑法：C:\\Python311\\python.exe -X utf8 D:\\orv_bench\\probe_attn_leaf.py
"""
import os
import sys

WORK = r"D:\orv_bench"
V3DIR = r"D:/TGAI-V3-1.5B/03_src"
os.chdir(WORK)
sys.path.insert(0, V3DIR)
sys.path.insert(0, WORK)

import torch                                                    # noqa: E402
from torch.profiler import profile, ProfilerActivity            # noqa: E402

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
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    step()
    torch.cuda.synchronize()

evs = list(prof.events())
leaf = [e for e in evs if not getattr(e, "cpu_children", ()) and
        float(getattr(e, "self_device_time_total", 0) or 0) > 0]
print("叶子合计 %.1f ms ；全事件 self_dev 合计 %.1f ms"
      % (sum(float(e.self_device_time_total) for e in leaf) / 1000.0,
         sum(float(getattr(e, "self_device_time_total", 0) or 0) for e in evs) / 1000.0))

for nm in ("aten::_efficient_attention_forward", "aten::_efficient_attention_backward"):
    for e in evs:
        if e.key != nm or float(getattr(e, "self_device_time_total", 0) or 0) <= 0:
            continue
        d = float(e.self_device_time_total) / 1000.0
        ch = list(getattr(e, "cpu_children", ()))
        ch_sum = sum(float(getattr(c, "self_device_time_total", 0) or 0)
                     for c in ch) / 1000.0
        print("%-42s self_dev=%.2f ms  cpu_children=%d  子 self_dev 合计=%.2f ms"
              % (nm, d, len(ch), ch_sum))
        for c in ch[:6]:
            print("      %-40s %.2f ms"
                  % (c.key, float(getattr(c, "self_device_time_total", 0) or 0) / 1000.0))
        break
