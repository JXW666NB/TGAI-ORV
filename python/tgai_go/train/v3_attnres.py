# -*- coding: utf-8 -*-
"""BlockAttnRes（V3 的 K3 Block Attention Residuals）的 ORV 实现。

参考实现：`model_v3.py:1227-1282`（严格照论文 arXiv:2603.15031 的伪代码）。

官方伪代码：
    V = stack(blocks + [partial_block])                  # [N+1, B, T, D]
    K = norm(V)                                          # 逐块 RMSNorm 等价
    logits = einsum('d, n b t d -> n b t', proj.w, K)    # [N+1, B, T]
    h = einsum('n b t, n b t d -> b t d', logits.softmax(0), V)   # [B, T, D]

**改写原因**：ORV 的 `einsum` 在 CUDA 上抛 NotImplementedError（`__init__.py:2711`）。
改写只换写法、不换数学：
    einsum('d,nbtd->nbt', w, K)  ->  (K.reshape(-1,D) @ w.reshape(D,1)).reshape(N+1,B,T)
    einsum('nbt,nbtd->btd', p, V) ->  (p.reshape(N+1,B*T,1) * V.reshape(N+1,B*T,D)).sum(0)
用到的都在内核里：matmul、softmax、sum(axis)、广播乘、stack。
"""
import numpy as np


def _as_list(blocks):
    return list(blocks)


def block_attn_res(blocks, partial_block, w, norm_w, eps=1e-6):
    """ORV 实现。`blocks`：已完成 block 的 [B,T,D] 张量列表；`partial_block`：当前部分和。

    `w`：(D,) 伪查询（对应 `proj.weight.reshape(-1)`）；`norm_w`：(D,) RMSNorm 的 γ。
    """
    import tgai_go.train as T
    from tgai_go.train import nn_ops

    blocks = _as_list(blocks)
    if not blocks:
        return partial_block
    D = int(partial_block.shape[-1])
    B, Tn = int(partial_block.shape[0]), int(partial_block.shape[1])

    # 逐块归一化（与"先 stack 再 norm"数学等价，但省一份 fp32 副本）
    Ks = [nn_ops.rms_norm(t, norm_w, eps) for t in list(blocks) + [partial_block]]
    Vs = list(blocks) + [partial_block]
    n = len(Vs)

    K = T.stack(Ks, dim=0)                      # [n, B, T, D]
    V = T.stack(Vs, dim=0)                      # [n, B, T, D]

    w2 = w.reshape(D, 1)                                   # (D,1)
    logits = (K.reshape(n * B * Tn, D) @ w2).reshape(n, B, Tn)   # [n,B,T]
    p = T.functional.softmax(logits, axis=0)               # 沿 block 维
    h = (p.reshape(n, B * Tn, 1) * V.reshape(n, B * Tn, D)).sum(axis=0)
    return h.reshape(B, Tn, D)


def block_attn_res_numpy(blocks, partial_block, w, norm_w, eps=1e-6):
    """参考实现（numpy，用于 CPU 对拍）。与 `block_attn_res` 同数学。"""
    blocks = _as_list(blocks)
    if not blocks:
        return np.asarray(partial_block, np.float32)
    Vs = [np.asarray(t, np.float32) for t in list(blocks) + [partial_block]]
    V = np.stack(Vs, 0)                                    # [n,B,T,D]
    K = V / np.sqrt((V * V).mean(-1, keepdims=True) + eps) * np.asarray(norm_w, np.float32)
    w = np.asarray(w, np.float32).reshape(-1)
    logits = np.einsum("d,nbtd->nbt", w, K)                # [n,B,T]
    e = np.exp(logits - logits.max(0, keepdims=True))
    p = e / e.sum(0, keepdims=True)
    return np.einsum("nbt,nbtd->btd", p, V)
