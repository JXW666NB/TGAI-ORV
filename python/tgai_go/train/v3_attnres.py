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

    Vs = list(blocks) + [partial_block]
    w2 = w.reshape(D, 1)                                   # (D,1)

    # ★ 不 stack：ORV 的 CUDA stack 未实现（__init__.py:2453 抛 NotImplementedError），
    #   CPU 上跑得通会把这个问题完全掩盖掉 —— 2026-11-16 在 V100 上实测到的。
    #   改成逐块算 logits，再用逐元素 max/exp/sum 做 softmax。
    logits = [(nn_ops.rms_norm(t, norm_w, eps).reshape(B * Tn, D) @ w2).reshape(B, Tn)
              for t in Vs]
    # softmax over blocks，逐元素完成。不减最大值：w 初值 std=0.02、norm 后量级很小，
    # exp 不会溢出；且减去同一常数对归一化后的结果没有影响（numpy 参考里减了，两者等价）。
    es = [lg.exp() for lg in logits]
    s = es[0]
    for e in es[1:]:
        s = s + e
    out = None
    for e, V_t in zip(es, Vs):
        term = (e / s).reshape(B, Tn, 1) * V_t
        out = term if out is None else out + term
    return out


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


def run_attn_res_layers(x, layers, attn_res, mlp_res, block_size=2):
    """BlockAttnRes 的整栈前向，严格照 model_v3.py:1356-1389 的循环。

    layers：长度 n_layers，每项是 (attn_part, mlp_part) 两个可调用对象，
            都接收 [B,T,D] 返回 [B,T,D]。
    attn_res / mlp_res：长度 n_layers，每项是 (w, norm_w) 二元组，
            分别用于 attn 前 / mlp 前的深度注意力。

    循环语义（官方伪代码，逐行照搬）：
        partial = x; blocks = []
        for i:
            h = attn_res[i](blocks, partial); a = attn_part(h)
            new_partial = partial + a
            if (i+1) % block_size == 0: blocks.append(new_partial)
            partial = new_partial
            h = mlp_res[i](blocks, partial); m = mlp_part(h)
            partial = partial + m
        return partial
    """
    partial = x
    blocks = []
    for i, (attn_part, mlp_part) in enumerate(layers):
        w, gw = attn_res[i]
        h = block_attn_res(blocks, partial, w, gw)
        a = attn_part(h)
        new_partial = partial + a
        if (i + 1) % int(block_size) == 0:
            blocks.append(new_partial)
        partial = new_partial
        w, gw = mlp_res[i]
        h = block_attn_res(blocks, partial, w, gw)
        partial = partial + mlp_part(h)
    return partial
