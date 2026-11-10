# -*- coding: utf-8 -*-
"""整段重写 conv.py 里 _conv2d_cuda_single / _conv2d_cuda / _cat_channels 三个函数。

背景：前几次编辑把单组函数的函数体弄丢了（只剩 `g = 1`），并把原体散落在别处。
本脚本用固定边界（`def _conv2d_cuda_single` → `def conv2d(`）整段替换，幂等。
"""
import ast
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "python", "tgai_go", "train", "conv.py")

BLOCK = '''def _conv2d_cuda_single(x, w, bias, stride, padding, dilation):
    """单组（groups=1）设备侧 conv2d：**不新写 CUDA 内核**，由 gather_flat（im2col）+ matmul 组合。

    依据（C3）：`gather_flat` 的前/反向已与 torch 逐位一致（0.00e+00），其反向是 atomicAdd
    scatter，正好是 conv 对输入梯度的语义；`_im2col_index_np` 的列序已用主机 `_im2col`
    逐配置比对（6 组含 padding/stride=2/dilation=2，全部 0.00e+00）。
    代价：多一份补丁矩阵（N*OH*OW*C*kh*kw），gather 逐元素，**大 kernel / 大 batch 明显慢于
    cuDNN**；索引在主机侧用 Python 循环构造。这是"可用且可验证"的实现，不是最优实现。
    """
    import tgai_go.train as _T
    from .indexing import gather_flat

    kh, kw = int(w.shape[2]), int(w.shape[3])
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    N, C, H, W = (int(v) for v in x.shape)
    Cout = int(w.shape[0])
    oh = _out_size(H, kh, sh, ph, dh)
    ow = _out_size(W, kw, sw, pw, dw)
    idx, msk = _im2col_index_np(N, C, H, W, kh, kw, sh, sw, ph, pw, dh, dw, oh, ow)
    cols = gather_flat(x, idx, n_src=N * C * H * W)
    if not msk.all():
        cols = cols * _T.tensor(np.ascontiguousarray(msk.astype(np.float32)))
    cols = cols.reshape(N * oh * ow, C * kh * kw)
    w2 = w.reshape(Cout, C * kh * kw).transpose(1, 0)          # [C*kh*kw, Cout]
    y = (cols @ w2).reshape(N, oh, ow, Cout).permute(0, 3, 1, 2)
    if bias is not None:
        y = y + _as_tensor(bias).reshape(1, Cout, 1, 1)
    return y


def _conv2d_cuda(x, w, bias, stride, padding, dilation, groups):
    """设备侧 conv2d 入口：groups=1 走单组；groups>1 按组切片后逐组算，再拼通道。

    组卷积在数学上就是"各组独立卷积 + 通道拼接"，所以复用已验证到 2.39e-07 的单组路径即可。
    """
    g = int(groups)
    if g <= 0:
        raise ValueError("groups 必须为正，收到 %r" % (groups,) + "。修法：常规卷积用 groups=1")
    if g == 1:
        return _conv2d_cuda_single(x, w, bias, stride, padding, dilation)

    N, C, H, W = (int(v) for v in x.shape)
    Cout = int(w.shape[0])
    if C % g or Cout % g:
        raise ValueError("groups=%d 与通道数不匹配（Cin=%d Cout=%d）" % (g, C, Cout) +
                         "。修法：groups 必须同时整除输入与输出通道数")
    Cg, Og = C // g, Cout // g
    outs = []
    for gi in range(g):
        xg = x[:, gi * Cg:(gi + 1) * Cg, :, :]
        wg = w[gi * Og:(gi + 1) * Og, :, :, :]
        outs.append(_conv2d_cuda_single(xg, wg, None, stride, padding, dilation))
    y = outs[0] if len(outs) == 1 else _cat_channels(outs)
    if bias is not None:
        y = y + _as_tensor(bias).reshape(1, Cout, 1, 1)
    return y


def _cat_channels(parts):
    """沿通道维（dim=1）拼接。注意 `tgai_go.train.cat` 的形参是 `dim`，不是 `axis`。"""
    import tgai_go.train as _T
    return _T.cat(parts, dim=1)


'''


def main():
    s = io.open(TARGET, encoding="utf-8").read()
    start = s.find("def _conv2d_cuda_single")
    end = s.find("def conv2d(x, weight")
    if start < 0 or end < 0 or end <= start:
        print("边界找不到：start=%d end=%d" % (start, end))
        return 2
    out = s[:start] + BLOCK + s[end:]
    ast.parse(out)
    io.open(TARGET, "w", encoding="utf-8", newline="\n").write(out)
    print("已整段重写单组/调度器/拼接三个函数")
    return 0


if __name__ == "__main__":
    sys.exit(main())
