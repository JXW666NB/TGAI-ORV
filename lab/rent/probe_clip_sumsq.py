# -*- coding: utf-8 -*-
"""定位 `clip_grad_norm_` 的 286 ms：单个 kernel 的微基准（CUDA event，不 sync 包裹）。

背景（本次实测）：seq512 / vocab96000 的一步里 `clip` 段 event 跨度 **285.9 ms**，
vocab4096 只有 **21.3 ms**；而 `clip_grad_norm_` 的代码看起来只是「2 次发射 + 读一个标量」
（`nn_ops.py:207-266`）。可疑点：`cuda.py:8321` 的 `sumsq_multi_slots` 与 `cuda.py:8340`
的 `scale_multi` 都用 **grid=(n_tensors,1,1)、block=256** —— 即**每个张量只给一个 block**。
对 (96000,1024)=98.3M 元素的梯度，一个 block 要独自扫 393 MB / 读写 786 MB。

本脚本用**真实形状**造出同一组梯度缓冲，分别计时：
  A) `sumsq_multi_total`：n=321（2 大 + 319 小）  ← 训练里真实的情形
  B) `sumsq_multi_total`：n=320（1 大 + 319 小）  ← 对应 --tie（少一个 98.3M）
  C) `sumsq_multi_total`：n=321（全小，无 vocab 大张量）← 对应 vocab=4096
  D) `sumsq_multi_slots` 单核（用 event 直接量）
  E) `sumsq_parallel(98.3M)`（老的两段式，最多 1024 block）+ 321 次循环的「老路径」总时长
  F) `scale_multi`（3 种组合）与 `scale_inplace` 逐张量循环
每个都报 GB/s（按读写字节数），以及「单块 256 线程」的隐含带宽。
"""
import ctypes
import json
import statistics
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()

BIG_N = 96000 * 1024          # 98,304,000 → 393 MB (fp32)
SMALL_N = 3328 * 1024         # 3,407,872  → 13.6 MB（真实里最大的"小"张量 = W1/W2/W3）
N_SMALL = 319
V4096_N = 4096 * 1024         # 4,194,304  → 16.8 MB


def ev():
    return rt.event_create()


def timed(fn, n=3):
    """墙钟 + 3 次取中位（fn 内部自带同步/读标量）。"""
    fn()                                    # 预热
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def kern_ms(name, grid, block, args, nbytes=None, n=3):
    """单次 kernel 的 GPU 时间（event 对，外面 sync 一次再批量读）。"""
    evs = []
    for _ in range(n):
        e0, e1 = ev(), ev()
        rt.event_record(e0)
        rt.launch(name, grid, block, args)
        rt.event_record(e1)
        evs.append((e0, e1))
    rt.sync()
    ts = [rt.event_elapsed_ms(a, b) for a, b in evs]
    for a, b in evs:
        rt.event_destroy(a)
        rt.event_destroy(b)
    ms = statistics.median(ts)
    return ms, (nbytes / 1e6 / ms if (nbytes and ms) else None)


print("== 造缓冲 ==", flush=True)
big1 = rt.full((BIG_N,), 1.0)
big2 = rt.full((BIG_N,), 1.0)
bigs = [big1, big2]
smalls = [rt.full((SMALL_N if i % 2 == 0 else 1 << 20,), 1.0) for i in range(N_SMALL)]


def demo(tag, bufs, nbytes_read, nbytes_rw=None):
    ms = timed(lambda: rt.sumsq_multi_total(bufs))
    out = {"tag": tag, "n_tensors": len(bufs), "ms": round(ms, 3),
           "GB/s_read": round(nbytes_read / 1e6 / ms, 2) if ms else None}
    if nbytes_rw:
        sms = timed(lambda: rt.scale_multi(bufs, 0.5))
        out["scale_ms"] = round(sms, 3)
        out["scale_GBps_rw"] = round(nbytes_rw / 1e6 / sms, 2) if sms else None
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return out


ro = 1
for i in range(N_SMALL):
    ro += (SMALL_N if i % 2 == 0 else 1 << 20)
RO_BIG2 = 2 * BIG_N + ro
RO_BIG1 = BIG_N + ro
RO_V4096 = 2 * V4096_N + ro
RW_BIG2 = 2 * (2 * BIG_N + ro)
RW_BIG1 = 2 * (BIG_N + ro)
RW_V4096 = 2 * (2 * V4096_N + ro)

res = []
res.append(demo("A_321_2big", bigs + smalls, 4 * RO_BIG2, 4 * RW_BIG2))
res.append(demo("B_320_1big", [big1] + smalls, 4 * RO_BIG1, 4 * RW_BIG1))
v4 = [rt.full((V4096_N,), 1.0), rt.full((V4096_N,), 1.0)]
res.append(demo("C_321_v4096", v4 + smalls, 4 * RO_V4096, 4 * RW_V4096))

print("\n== D) 单核 grid=(n,1,1) block=256 的实测 ==", flush=True)
bufs = bigs + smalls
n = len(bufs)
slots = rt.empty((n,), dtype=np.float32)
ptrs, sizes = rt._ptrs_and_sizes(bufs)
ms, gbs = kern_ms("sumsq_multi_slots", (n, 1, 1), (256, 1, 1),
                  [rt._ptr(ptrs), rt._ptr(sizes), (n, ctypes.c_int),
                   rt._ptr(slots)], nbytes=4 * RO_BIG2)
print(json.dumps({"kernel": "sumsq_multi_slots", "grid": [n, 1, 1], "block": 256,
                  "ms": round(ms, 3), "GB/s": round(gbs, 2)}, ensure_ascii=False), flush=True)
ms2, gbs2 = kern_ms("sumsq_final", (1, 1, 1), (256, 1, 1),
                    [rt._ptr(slots), rt._ptr(rt.empty((), dtype=np.float32)),
                     (n, ctypes.c_int)])
print(json.dumps({"kernel": "sumsq_final", "ms": round(ms2, 3)}, ensure_ascii=False), flush=True)

# 只保留一个大张量（模拟 --tie）
bufs1 = [big1] + smalls
n1 = len(bufs1)
slots1 = rt.empty((n1,), dtype=np.float32)
p1, s1 = rt._ptrs_and_sizes(bufs1)
ms1b, g1b = kern_ms("sumsq_multi_slots", (n1, 1, 1), (256, 1, 1),
                    [rt._ptr(p1), rt._ptr(s1), (n1, ctypes.c_int), rt._ptr(slots1)],
                    nbytes=4 * RO_BIG1)
print(json.dumps({"kernel": "sumsq_multi_slots", "case": "1 big（tie）",
                  "ms": round(ms1b, 3), "GB/s": round(g1b, 2)}, ensure_ascii=False), flush=True)

# scale_multi 单核
n3 = len(bufs)
p3, s3 = rt._ptrs_and_sizes(bufs)
ms3, g3 = kern_ms("scale_multi", (n3, 1, 1), (256, 1, 1),
                  [rt._ptr(p3), rt._ptr(s3), (n3, ctypes.c_int), (0.5, ctypes.c_float)],
                  nbytes=4 * RW_BIG2)
print(json.dumps({"kernel": "scale_multi", "grid": [n3, 1, 1], "block": 256,
                  "ms": round(ms3, 3), "GB/s": round(g3, 2)}, ensure_ascii=False), flush=True)

print("\n== E) 老路径（每张量 sumsq_parallel，最多 1024 block）==", flush=True)
ms4, g4 = kern_ms("sumsq_partial", (1024, 1, 1), (256, 1, 1),
                  [rt._ptr(big1), rt._ptr(rt.empty((1024,), dtype=np.float32)),
                   (BIG_N, ctypes.c_int)], nbytes=4 * BIG_N)
print(json.dumps({"kernel": "sumsq_partial", "n": BIG_N, "grid": [1024, 1, 1],
                  "ms": round(ms4, 3), "GB/s": round(g4, 2)}, ensure_ascii=False), flush=True)


def old_path():
    acc = None
    for b in bufs:
        ss = rt.sumsq_parallel(b)
        acc = ss if acc is None else rt.add_(acc, ss)
    return float(rt.read_scalar(acc))


ms5 = timed(old_path)
print(json.dumps({"path": "old_321x_sumsq_parallel(+add_)", "ms": round(ms5, 3),
                  "GB/s_read": round(4 * RO_BIG2 / 1e6 / ms5, 2)}, ensure_ascii=False), flush=True)


def old_scale():
    for b in bufs:
        rt.scale_inplace(b, 0.5)


ms6 = timed(old_scale)
print(json.dumps({"path": "old_321x_scale_inplace", "ms": round(ms6, 3),
                  "GB/s_rw": round(4 * RW_BIG2 / 1e6 / ms6, 2)}, ensure_ascii=False), flush=True)
