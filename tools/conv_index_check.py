# -*- coding: utf-8 -*-
"""用主机侧 `_im2col` 作为基准，验证 `_im2col_index_np` 的列序一致（含 padding/stride/dilation）。

基准是**主机实现本身**，不是我的理解：`_im2col` 返回 (cols, (oh, ow))，这里逐配置比 cols。
CPU-only，可在本机跑：python tools/conv_index_check.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\DSH-Workspace\TGAI-对话档案\_tgai_orv\python")

from tgai_go.train.conv import _im2col, _im2col_index_np


def check(x, kh, kw, sh, sw, ph, pw, dh, dw):
    N, C, H, W = x.shape
    cols, (oh, ow) = _im2col(x, kh, kw, sh, sw, ph, pw, dh, dw)
    idx, msk = _im2col_index_np(N, C, H, W, kh, kw, sh, sw, ph, pw, dh, dw, oh, ow)
    flat = x.reshape(-1).astype(np.float32)
    got = flat[idx].reshape(cols.shape)
    if not msk.all():
        got = got * msk.astype(np.float32).reshape(cols.shape)
    ok = np.array_equal(got, cols)
    worst = float(np.abs(got - cols).max())
    print("kh=%d kw=%d s=%d,%d p=%d,%d d=%d,%d -> oh,ow=%s  一致=%s  最大差=%.2e"
          % (kh, kw, sh, sw, ph, pw, dh, dw, (oh, ow), ok, worst))
    return ok


def main():
    rng = np.random.default_rng(0)
    total_ok = True
    cfgs = [
        # (N,C,H,W, kh,kw, sh,sw, ph,pw, dh,dw)
        (2, 3, 5, 6, 3, 3, 1, 1, 0, 0, 1, 1),
        (1, 2, 6, 6, 3, 3, 1, 1, 1, 1, 1, 1),
        (2, 2, 7, 7, 3, 3, 2, 2, 1, 1, 1, 1),
        (1, 3, 8, 5, 5, 3, 1, 1, 2, 1, 1, 1),
        (1, 1, 9, 9, 3, 3, 1, 1, 2, 2, 2, 2),
        (2, 4, 5, 5, 2, 2, 2, 1, 0, 1, 1, 1),
    ]
    for (N, C, H, W, kh, kw, sh, sw, ph, pw, dh, dw) in cfgs:
        x = rng.normal(0, 1, (N, C, H, W)).astype(np.float32)
        total_ok = check(x, kh, kw, sh, sw, ph, pw, dh, dw) and total_ok
    print("总结论:", "PASS" if total_ok else "FAIL")
    return 0 if total_ok else 1


if __name__ == "__main__":
    sys.exit(main())
