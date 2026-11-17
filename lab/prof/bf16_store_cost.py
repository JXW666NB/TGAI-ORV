# -*- coding: utf-8 -*-
"""bf16 存储的成本实测：每层激活存/取各一次 cast，一步要花多少时间？

模型实测（1.5B / 24 层 / seq512）：
  · 每层保留的激活约 0.38 GiB ⇒ 单张量 (T*d) = 512*1024 = 524288 元素
  · 一层里需要"存+取"的大约 3 个张量（ln1 输出 / attn 输出 / mlp 输入）
本脚本量这三次 cast 的真实耗时，再乘以 24 层，得到每步新增开销的下限。
"""
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)

SEQ, D, LAYERS, PER_LAYER = 512, 1024, 24, 3


def timed(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    rt.sync()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    n = SEQ * D
    x = rt.h2d(rng.normal(0, 1.0, (n,)).astype(np.float32))

    def store():
        return rt._cast_flat_ptr_bf16(int(x.ptr), n)

    ms_store = timed(store)
    # 取回来：bf16 -> fp32（用同一个 kernel 家族；没有专门 wrapper 时按同量级估）
    b = store()
    ms_load = timed(lambda: rt._cast_flat_ptr(int(b.ptr), n, False)) \
        if hasattr(rt, "_cast_flat_ptr") else float("nan")

    per_layer = ms_store + (0.0 if ms_load != ms_load else ms_load)
    print("=== bf16 存储/读取成本（n=%d = seq%d×d%d）===" % (n, SEQ, D))
    print("  存 fp32→bf16 : %8.4f ms" % ms_store)
    print("  取 bf16→fp32 : %s" % ("—（无 wrapper，跳过）" if ms_load != ms_load
                                   else "%8.4f ms" % ms_load))
    print("  每层(%d 个张量) : %8.4f ms" % (PER_LAYER, per_layer * PER_LAYER))
    print("  %d 层合计     : %8.2f ms / 步" % (LAYERS, per_layer * PER_LAYER * LAYERS))
    print("\n  对照：1.5B/seq512 当前一步 ≈ 510 ms")
    tot = per_layer * PER_LAYER * LAYERS
    print("  ⇒ 预计涨幅 ≈ %.1f%%" % (tot / 510.0 * 100.0))
    print("\n  数学事实：bf16 尾数 8 位 ⇒ 相对舍入误差最坏 2^-9 = %.2f%%"
          % (2 ** -9 * 100))


if __name__ == "__main__":
    main()
