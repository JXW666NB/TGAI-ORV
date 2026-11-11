# -*- coding: utf-8 -*-
"""给 SDPA 加 torch 风格 4 维输入支持（用包装器，不改内部实现）。

## 为什么用包装器而不是改内部

P100 用户照 torch 习惯传 `[B, H, S, Dh]`，ORV 内部要求 `[B*H, S, Dh]`。
如果在内部加 `_was4d` 标志再在各条 return 上还原，只要漏一处就会**静默返回错形状**——
这种 bug 比不支持更糟（形状错了还会一路算下去）。

所以：把原实现改名 `_sdpa_impl`，新加一层薄包装，展平 → 调用 → 还原。
还原只有一处，不会被漏掉。
"""
import ast
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "python", "tgai_go", "train", "__init__.py")
s = io.open(P, encoding="utf-8").read()

SIG = "def scaled_dot_product_attention("
if "def _sdpa_impl(" in s:
    print("已打过补丁，跳过")
    sys.exit(0)

i = s.find(SIG)
if i < 0:
    print("找不到 scaled_dot_product_attention")
    sys.exit(1)

# 定位方法定义的缩进与起始
line_start = s.rfind("\n", 0, i) + 1
indent = s[line_start:i]
print("缩进为 %r" % indent)

WRAPPER = '''{ind}def scaled_dot_product_attention(q, k, v, causal=False, is_causal=None,
{ind}                               scale=None, lowmem=False):
{ind}    """与 torch 的 `F.scaled_dot_product_attention` 兼容的入口。

{ind}    两种输入都可：
{ind}      * `[B*H, S, Dh]` —— 内部展平形式（原有用法，保持不变）
{ind}      * `[B, H, S, Dh]` —— **torch 的写法**，自动展平并在返回时还原

{ind}    其它参数与 `_sdpa_impl` 完全一致。
{ind}    """
{ind}    rank = len(tuple(q.shape))
{ind}    if rank != 4:
{ind}        return _sdpa_impl(q, k, v, causal=causal, is_causal=is_causal,
{ind}                          scale=scale, lowmem=lowmem)
{ind}    B, H, S, Dh = (int(q.shape[0]), int(q.shape[1]),
{ind}                    int(q.shape[2]), int(q.shape[3]))
{ind}    qf = q.reshape(B * H, S, Dh)
{ind}    kf = k.reshape(int(k.shape[0]) * int(k.shape[1]), int(k.shape[2]),
{ind}                      int(k.shape[3]))
{ind}    vf = v.reshape(int(v.shape[0]) * int(v.shape[1]), int(v.shape[2]),
{ind}                      int(v.shape[3]))
{ind}    out = _sdpa_impl(qf, kf, vf, causal=causal, is_causal=is_causal,
{ind}                     scale=scale, lowmem=lowmem)
{ind}    return out.reshape(B, H, S, Dh)

'''.format(ind=indent)

# 1) 原函数改名
s = s[:i] + s[i:].replace(SIG, "def _sdpa_impl(", 1)
# 2) 在改名后的定义前插入包装器
j = s.find("def _sdpa_impl(")
jline = s.rfind("\n", 0, j) + 1
s = s[:jline] + WRAPPER + s[jline:]

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
try:
    ast.parse(s)
    print("补丁已应用，语法 OK")
except SyntaxError as ex:
    print("语法错误:", ex)
    sys.exit(1)
