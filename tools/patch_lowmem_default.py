# -*- coding: utf-8 -*-
"""把 SDPA 的 lowmem 默认值从 False 改成「自动（None）」。

背景（2026-11-15 真机 P100 定位）：`bench_p100_full.py:149` 硬编码 lowmem=False，
理由是「lowmem 在 sm_60 需要 68 KB smem > 48 KB 上限」。但那个限制已经被修掉
（滑窗反向只要 40,448 B）。结果 benchmark 一直在测物化注意力：
峰值 11.33 GB；改成 lowmem=True 后 7.14 GB（比 torch 的 7.37 GB 还低）。

所以默认值也不该是 False：P100 上完全可用，默认关掉只会让调用方踩坑。
改成 None = 自动：_sdpa_lowmem_ok 会按 dh==64 / causal / CUDA / 内核可用性判断，
不满足就自动回退物化（原行为）。
"""
import io

P = "python/tgai_go/train/__init__.py"

OLD_SIG = """def scaled_dot_product_attention(q, k, v, causal=False, is_causal=None,
                                 scale=None, lowmem=False, attn_mask=None):"""

NEW_SIG = """def scaled_dot_product_attention(q, k, v, causal=False, is_causal=None,
                                 scale=None, lowmem=None, attn_mask=None):"""

INSERT = """
        if lowmem is None:
            # ★ 2026-11-15：默认改为「自动」。以前默认 False，而 benchmark 也照着
            #   老限制（sm_60 需 68 KB smem）硬编码关掉，导致 P100 明明能用 39.5 KB
            #   的滑窗融合反向却一直走物化注意力：实测 11.33 GB -> 7.14 GB。
            #   现在默认尝试融合，_sdpa_lowmem_ok 会按 dh==64 / causal / CUDA /
            #   内核可用性判断，不满足自动回退物化。
            lowmem = True
"""


def main():
    s = io.open(P, encoding="utf-8").read()
    if "lowmem=None, attn_mask=None" in s:
        print("已经改过")
        return 0
    assert OLD_SIG in s, "没找到函数签名"
    s = s.replace(OLD_SIG, NEW_SIG, 1)

    # 在 docstring 之后、函数体第一条语句之前插入
    doc_start = s.index(NEW_SIG)
    doc_q = s.index('"""', doc_start)
    doc_end = s.index('"""', doc_q + 3) + 3
    s = s[:doc_end] + INSERT + s[doc_end:]

    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    import ast
    ast.parse(io.open(P, encoding="utf-8").read())
    print("已改：lowmem 默认 None（自动）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
