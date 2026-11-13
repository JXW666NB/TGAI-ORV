# -*- coding: utf-8 -*-
"""分块因果注意力：把 `B·H·S²` 的分数矩阵降成 `B·H·C·S`（C=块大小）。

## 为什么这样能省显存又不必写 CUDA

物化路径的峰值来自 `scores = q·kᵀ`，形状 `B·H·S·S`，`S²` 直接进显存。
把 query 切成 `S/C` 块，每块只与"截至本块末尾"的 key 相乘：

    块 i：q = [iC, (i+1)C)，k = [0, (i+1)C)   ⇒ 分数 B·H·C·(i+1)C ≤ B·H·C·S

于是峰值从 `B·H·S²` 降到 `B·H·C·S`，S=2048、C=256 时是 8× 的削减。
反向由 ORV 的 autograd 顺着这些 SDPA 调用自动完成，不需要自定义内核。

## 正确性要点

块 i 的掩码不是标准方阵因果：行 r 是绝对位置 `iC+r`，key 是绝对位置 `0..(i+1)C-1`，
所以可见条件是 `k ≤ iC + r`，用**矩形加性掩码**逐块构造（主机侧构造，代价可忽略）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))


def _rect_mask(rows, cols, q_offset):
    """块掩码：行绝对位置 = q_offset + r，可见 key ≤ 该位置。形状 (1,rows,cols)。"""
    r = np.arange(rows)[:, None] + int(q_offset)
    c = np.arange(cols)[None, :]
    return np.where(c <= r, 0.0, -np.inf).astype(np.float32)[None, :, :]


def attention_orv_chunked(x, wq, wk, wv, wo, cos, sin, rot_dim, n_q, n_kv,
                          chunk=256, causal=True):
    """与 `attention_orv(..., causal=True)` 等价，但按 query 块计算以省显存。"""
    import tgai_go.train as T
    from tgai_go.train import v3_ops as V

    B = int(x.shape[0])
    Tlen = int(x.shape[1])
    n_q, n_kv = int(n_q), int(n_kv)
    d_k = int(wq.shape[0]) // n_q

    q_all = (x @ wq.transpose(0, 1)).reshape(B, Tlen, n_q, d_k).transpose(1, 2)
    k_all = (x @ wk.transpose(0, 1)).reshape(B, Tlen, n_kv, d_k).transpose(1, 2)
    v_all = (x @ wv.transpose(0, 1)).reshape(B, Tlen, n_kv, d_k).transpose(1, 2)

    c = cos.reshape(1, 1, Tlen, int(rot_dim))
    s = sin.reshape(1, 1, Tlen, int(rot_dim))
    q_all = V.v3_rope(q_all, c, s)
    k_all = V.v3_rope(k_all, c, s)

    if n_kv != n_q:
        rep = n_q // n_kv
        k_all = T.cat([k_all[:, h:h + 1] for h in range(n_kv) for _ in range(rep)], dim=1)
        v_all = T.cat([v_all[:, h:h + 1] for h in range(n_kv) for _ in range(rep)], dim=1)

    outs = []
    C = max(1, int(chunk))
    for start in range(0, Tlen, C):
        end = min(start + C, Tlen)
        qb = q_all[:, :, start:end, :]
        kb = k_all[:, :, :end, :]                    # 只到本块末尾，未来的 key 根本不参与
        vb = v_all[:, :, :end, :]
        m = _rect_mask(end - start, end, start)      # (1,rows,cols)
        am = np.repeat(m, n_q * B, axis=0).reshape(B * n_q, end - start, end)
        ob = T.functional.scaled_dot_product_attention(
            qb, kb, vb, causal=False, attn_mask=T.tensor(np.ascontiguousarray(am)))
        outs.append(ob)

    out = outs[0] if len(outs) == 1 else T.cat(outs, dim=2)
    out = out.transpose(1, 2).reshape(B, Tlen, n_q * d_k)
    return out @ wo.transpose(0, 1)
