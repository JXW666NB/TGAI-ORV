# -*- coding: utf-8 -*-
"""单点验证：`Tensor.compact_bf16()` 机制是否真的省显存、且读回来数值正确。

只验机制（一个张量），不掺进模型 —— 因为要真正省下"前向产出→反向读取"这段的显存，
还需要让算子的反向闭包**捕获 Tensor 而不是缓冲**（下一步再做）。

判据：
  1. 压缩后池借出字节应**减少约一半**；
  2. 读完（升回 fp32）后字节回到原量级（临时缓冲，随后释放）；
  3. 往返数值误差 ≤ 2^-9 ≈ 1.95e-3（bf16 八位尾数的上界）。
"""
import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()


def live_bytes():
    return rt.pool_dump_live(top=1)["bytes"]


def main():
    gi = float(1 << 30)
    n = 1024 * 1024
    ref = np.random.default_rng(0).normal(0, 0.05, (n,)).astype(np.float32)

    base = live_bytes()
    x = T.tensor(ref)
    after_alloc = live_bytes()
    print("  分配后       : +%.2f MB" % ((after_alloc - base) / (1 << 20)))

    x.compact_bf16()
    after_compact = live_bytes()
    print("  compact_bf16 : %+.2f MB（相对分配后，应为负的一半）"
          % ((after_compact - after_alloc) / (1 << 20)))
    print("  is_bf16_stored = %s" % x.is_bf16_stored())

    # 读一次：升回 fp32，做一次简单运算
    back = x._gpu
    d = np.asarray(rt.d2h(back)).reshape(-1)
    max_rel = float(np.abs(d - ref).max() / max(1e-30, np.abs(ref).max()))
    print("  往返 max_rel = %.3e   （上界 2^-9 = %.3e）" % (max_rel, 2.0 ** -9))

    after_read = live_bytes()
    print("  读一次后     : %+.2f MB（临时 fp32 还在）" % ((after_read - after_compact) / (1 << 20)))
    del back
    rt.sync()
    after_del = live_bytes()
    print("  释放读结果后 : %+.2f MB" % ((after_del - after_read) / (1 << 20)))

    ok_mem = (after_compact - after_alloc) < 0
    ok_num = max_rel <= 2.0 ** -9
    print("\n  判定：省显存 %s；数值 %s" % ("OK" if ok_mem else "**未生效**",
                                          "OK" if ok_num else "**超界**"))
    # 再验一个真实算子：用压缩过的张量做 2-D matmul，看与直接 fp32 的差
    m = 64
    ref2 = ref.reshape(m, n // m)
    x2 = T.tensor(ref2)
    x2.compact_bf16()
    w = T.tensor(np.random.default_rng(1).normal(
        0, 0.05, (n // m, 128)).astype(np.float32))
    y1 = T.tensor(ref2) @ w
    y2 = x2 @ w
    a = np.asarray(rt.d2h(y1._gpu))
    b = np.asarray(rt.d2h(y2._gpu))
    rel = float(np.abs(a - b).max() / max(1e-30, np.abs(a).max()))
    print("  经 matmul 的相对差 = %.3e（应 ≲2^-9）" % rel)


if __name__ == "__main__":
    main()
