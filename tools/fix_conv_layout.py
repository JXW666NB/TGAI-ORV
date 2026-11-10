# -*- coding: utf-8 -*-
"""修复 conv.py 的布局：把 _cat_channels 从单组函数中间移到函数之后，并加回分组调度器。

幂等：若已存在 `def _conv2d_cuda(x, w, bias, stride, padding, dilation, groups)` 则跳过。
"""
import ast
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "python", "tgai_go", "train", "conv.py")

DISPATCH = '''def _conv2d_cuda(x, w, bias, stride, padding, dilation, groups):
    """设备侧 conv2d 入口：groups=1 走单组路径；groups>1 按组切片后逐组算再拼通道。

    组卷积在数学上就是"各组独立卷积 + 通道拼接"，复用已验证到 2.39e-07 的单组路径即可。
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


'''


def main():
    s = io.open(TARGET, encoding="utf-8").read()
    if re.search(r"(?m)^def _conv2d_cuda\(x, w, bias, stride, padding, dilation, groups\):", s):
        print("已有分组调度器，跳过")
        return 0

    # 1) 取出 _cat_channels 那一块（到下一个顶层 def 之前）
    m = re.search(r"(?m)^def _cat_channels\(parts\):\n", s)
    if not m:
        print("找不到 _cat_channels")
        return 2
    nxt = re.search(r"(?m)^def ", s[m.end():])
    if not nxt:
        print("找不到 _cat_channels 的结束位置")
        return 2
    block_start = m.start()
    block_end = m.end() + nxt.start()
    cat_block = s[block_start:block_end]
    s_wo = s[:block_start] + s[block_end:]

    # 2) 在 `def conv2d(` 之前插入「调度器 + _cat_channels」
    i = s_wo.find("def conv2d(x, weight")
    if i < 0:
        print("找不到 conv2d 定义")
        return 2
    s_new = s_wo[:i] + DISPATCH + cat_block.rstrip() + "\n\n\n" + s_wo[i:]
    ast.parse(s_new)
    io.open(TARGET, "w", encoding="utf-8", newline="\n").write(s_new)
    print("已重排：单组函数 → 分组调度器 → _cat_channels → conv2d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
