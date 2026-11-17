# -*- coding: utf-8 -*-
"""验证：ORV 的 `transpose` / `permute` 是视图还是**真拷贝**？

如果 `w.transpose(0,1)` 会新分配一块 (in,out) 缓冲，那么每次 `x @ w.T` 前向
都会多留一份与权重同尺寸的拷贝，而反向的闭包又会把它攥住
⇒ 显存里每个权重活 3~4 份（retention.py 实测：4 层上 (2816,1024) 有 120 个、
(96000,1024) 有 6 个）。这份脚本直接在显存占用上验证。
"""
import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
GB = type(rt.zeros((1,), dtype=np.float32))

REG = {}
_o_init = GB.__init__
_o_del = GB.__del__


def g_init(self, *a, **kw):
    _o_init(self, *a, **kw)
    try:
        REG[id(self)] = (tuple(int(s) for s in self.shape), int(self.n) * 4)
    except Exception:
        pass


def g_del(self):
    REG.pop(id(self), None)
    return _o_del(self)


GB.__init__ = g_init
GB.__del__ = g_del


def live():
    return len(REG), sum(b for _s, b in REG.values()) / float(1 << 30)


def main():
    print("=== transpose / permute 是视图还是拷贝？===")
    n0, g0 = live()
    w = rt.h2d(np.zeros((2816, 1024), np.float32))
    n1, g1 = live()
    print("  h2d (2816,1024)        : +%d 个 / +%.4f GiB" % (n1 - n0, g1 - g0))

    t = w.transpose(0, 1)
    n2, g2 = live()
    print("  w.transpose(0,1)       : +%d 个 / +%.4f GiB  ← 非 0 就是**拷贝**"
          % (n2 - n1, g2 - g1))
    print("    结果形状 %s，是否同一块显存 ptr: %s"
          % (tuple(int(s) for s in t.shape), int(t.ptr) == int(w.ptr)))

    # 再看 CPU 侧张量（模型里权重是 T.tensor，走 _gpu）
    x = T.tensor(np.zeros((512, 1024), np.float32))
    n3, g3 = live()
    y = x @ T.tensor(np.zeros((2816, 1024), np.float32)).transpose(0, 1)
    n4, g4 = live()
    print("  x @ w.T                : +%d 个 / +%.4f GiB（含转置拷贝与 GEMM 输出）"
          % (n4 - n3, g4 - g3))

    # 显式看 transpose 后原权重是否还在、拷贝有多大
    n5, g5 = live()
    print("\n  当前活缓冲 %d 个 / %.4f GiB" % (n5, g5))
    big = sorted(((b, s) for s, b in REG.values()), reverse=True)[:6]
    for b, s in big:
        print("      %-26s %8.2f MB" % (str(s), b / (1 << 20)))


if __name__ == "__main__":
    main()
