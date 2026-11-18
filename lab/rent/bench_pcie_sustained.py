# -*- coding: utf-8 -*-
"""持续负载下的 PCIe 带宽：看链路会不会从 gen1 升档。

背景：一次性短拷贝测得 2.15 GiB/s，而当时 pcie.link.gen.current = 1（最大 4）。
GPU 空闲会把 PCIe 降到 gen1，短基准测到的就是降档速度，不能代表训练时的真实带宽。
这里连续跑 10 轮，并在负载中采样链路速率。
"""
import subprocess
import threading
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()


def link():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines()
    return " | ".join(s.strip() for s in out)


N = 848_645_257                       # 12 层 V3 的参数量 = 3.16 GiB
gb = N * 4 / float(1 << 30)
buf = rt.empty((N,), dtype=np.float32)
arr = np.zeros(N, dtype=np.float32)
_ = rt.d2h(buf)
_ = rt.h2d(arr)

print("开始链路：%s" % link())
stop = {"v": False}
seen = []


def sampler():
    while not stop["v"]:
        seen.append(link())
        time.sleep(0.4)


th = threading.Thread(target=sampler, daemon=True)
th.start()

for i in range(10):
    t0 = time.perf_counter()
    _ = rt.d2h(buf)
    t1 = time.perf_counter()
    b2 = rt.h2d(arr)
    t2 = time.perf_counter()
    print("  第 %2d 轮  d2h %5.2f GiB/s   h2d %5.2f GiB/s   链路 %s"
          % (i + 1, gb / (t1 - t0), gb / (t2 - t1), link()), flush=True)
    del b2

stop["v"] = True
th.join(timeout=2)
uniq = sorted(set(seen))
print("\n负载中采样到的链路状态（去重）：")
for s in uniq:
    print("  " + s)
print("结束链路：%s" % link())
