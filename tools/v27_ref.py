# -*- coding: utf-8 -*-
"""V2.7（3.0B MoE）的 **torch 参考前向** —— 作为 ORV 前向的对齐基准（CPU 可跑）。

语义全部有实测/源码依据，见 docs/V27_MOE_PORT.md：
* RMSNorm(eps=1e-6) → attn(q/k/v 分离、head_dim=64、**interleaved RoPE**) → out_proj → 残差
* RMSNorm → MoE：`z = x@router.T + qb_bias`（**top-k 之前**，`tgai_engine.c:1120`）→ topk(2) → softmax → 专家 SwiGLU → 残差
* `TIE=True`（无独立 lm_head，用 token_embedding 转置做输出）
* RoPE：θ=1e4、dh=64、j=0..31，对**相邻维度对** (2j,2j+1) 旋转（由预算表实测：cos[t,2j]==cos[t,2j+1]，前半≠后半）

`softmax` 的**作用范围**有两种自洽写法（源码只说"在 softmax 之前加偏置"）：
`smax="topk"`（只在选中的 2 个上归一化）或 `smax="all"`（全部 8 个归一化后取前 2）。
两者都保留，等与 ORV/产线 logits 比对后定夺，**不靠猜**。
"""
import math

import torch
import torch.nn.functional as F


def rmsnorm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_tables(seq, dh, theta=10000.0, device=None):
    """cos/sin 形状 [seq, dh]，相邻维度对共享频率（interleaved）。"""
    j = torch.arange(0, dh, 2, device=device, dtype=torch.float32) / dh
    inv = theta ** (-j)                       # [dh/2]
    t = torch.arange(seq, device=device, dtype=torch.float32)
    ang = t[:, None] * inv[None, :]           # [seq, dh/2]
    ang = torch.repeat_interleave(ang, 2, dim=1)   # 每频率给 2 个维度
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    """x: [seq, heads, dh]；相邻对旋转（interleaved）。"""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    c = cos[:, None, 0::2]
    s = sin[:, None, 0::2]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


def moe(x, p, cfg, smax="topk"):
    """x: [T, D] → [T, D]。p 是前缀参数字典（含 router/qb_bias/experts）。"""
    E, K = cfg["n_experts"], cfg["n_activated"]
    z = x @ p["router"].T
    if p.get("qb_bias") is not None:
        z = z + p["qb_bias"]
    top, idx = torch.topk(z, K, dim=-1)                 # 大值优先
    if smax == "topk":
        w = F.softmax(top, dim=-1)
    elif smax == "all":
        w = F.softmax(z, dim=-1).gather(1, idx)
    else:
        raise ValueError("smax 只支持 'topk' / 'all'（收到 %r）。修法：二选一，别自己发明"
                         % (smax,))
    out = torch.zeros_like(x)
    for k in range(K):
        ids = idx[:, k]
        wk = w[:, k:k + 1]
        for e in range(E):
            m = ids == e
            if not bool(m.any()):
                continue
            xe = x[m]
            g = xe @ p["experts"][e]["gate"].T
            u = xe @ p["experts"][e]["up"].T
            h = F.silu(g) * u
            out[m] += wk[m] * (h @ p["experts"][e]["down"].T)
    return out


def forward(tokens, P, cfg, smax="topk"):
    """tokens: [T] int64；P: {'embed','final_norm','blocks':[...]}。返回 logits [T, V]。"""
    D, NH = cfg["d_model"], cfg["n_heads"]
    DH = cfg["head_dim"]
    x = P["embed"][tokens]                       # [T, D]
    cos, sin = rope_tables(len(tokens), DH, cfg["rope_theta"])
    for i, b in enumerate(P["blocks"]):
        h = rmsnorm(x, b["ln1"])
        q = (h @ b["q"].T).view(-1, NH, DH)
        k = (h @ b["k"].T).view(-1, NH, DH)
        v = (h @ b["v"].T).view(-1, NH, DH)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1),
                                           v.transpose(0, 1), is_causal=True)
        a = a.transpose(0, 1).reshape(len(tokens), D)
        x = x + a @ b["out"].T
        h2 = rmsnorm(x, b["ln2"])
        x = x + moe(h2, b["moe"], cfg, smax=smax)
    x = rmsnorm(x, P["final_norm"])
    return x @ P["embed"].T                      # TIE=True