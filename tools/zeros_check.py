# -*- coding: utf-8 -*-
"""阻塞项验证：`rt.zeros` / `rt.full` 是否真的清零池化内存？计时是否可信？

不需要加载模型，几十秒即可。
运行：python zeros_check.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")


def main():
    import torch
    import tgai_go.train as T

    T.set_default_device("cuda")
    rt = T._cuda()
    n = 32 * 1024 * 1024            # 128 MB
    mb = n * 4 / 1024 ** 2

    # 1) 先填 1，再填 0（同一尺寸，池很可能复用同一块）
    a = rt.full((n,), 1.0)
    va = np.asarray(rt.d2h(a), np.float32)
    b = rt.full((n,), 0.0)
    vb = np.asarray(rt.d2h(b), np.float32)
    print("full(1.0) 取值集合: %s" % np.unique(va)[:4])
    print("随后 full(0.0) 取值集合: %s" % np.unique(vb)[:4])
    dirty = bool(np.any(vb != 0.0))
    print("=> 清零是否真的发生: %s" % ("否（发现非零残留）" if dirty else "是"))

    # 2) 计时：20 次 128 MB 填充（整设备同步）
    rt.full((n,), 0.0); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        rt.full((n,), 0.0)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 20
    print("20 次填充 %.0f MB：%.3f ms/次 ⇒ 有效带宽 %.0f GB/s（V100 上限约 900）"
          % (mb, dt * 1000, (n * 4) / max(dt, 1e-9) / 1e9))

    # 3) 对照：一次真正的 H2D（128 MB），这条路径一定计入设备时间
    host = np.ones(n, np.float32)
    t0 = time.time()
    for _ in range(5):
        buf = rt.h2d(host)
    torch.cuda.synchronize()
    dt2 = (time.time() - t0) / 5
    print("h2d %.0f MB：%.2f ms/次 ⇒ %.0f GB/s" % (mb, dt2 * 1000, (n * 4) / max(dt2, 1e-9) / 1e9))
    print("结论：若 (2) 远小于 (3) 的物理下界，则 (2) 的计时没有覆盖真实内存工作。")


if __name__ == "__main__":
    main()
