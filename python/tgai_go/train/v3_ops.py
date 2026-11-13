# -*- coding: utf-8 -*-
"""V3 算子移植（第一批）：SiTU-GLU。

## 参考实现（`D:\\TGAI-V3-1.5B\\03_src\\model_v3.py` 的 `SiTUGLUFFN.forward`）

```
g = w1(x); u = w2(x)
gate = (beta1 * tanh(g / beta1)) * sigmoid(g)
up   = beta2 * tanh(u / beta2)
y    = w3(gate * up)
```

β₁ = 4（gate 分支）、β₂ = 25（up 分支）⇒ 输出上界 |f| ≤ β₁·β₂ = 100。
论文动机：SwiGLU 的两个乘子都**无界**，低精度下容易溢出；SiTU 用 tanh 把乘积硬压住，
而 tanh 在原点近似线性 ⇒ 局部行为贴近 SwiGLU，不损失表达能力。

## 为什么这里不直接用 `S.sigmoid`

ORV 的 `train` 包里 **没有** 导出 `sigmoid`（只确认了 `tanh`）。用恒等式

    sigmoid(x) = 0.5 * (1 + tanh(x / 2))

实现，数学上**精确相等**，且只依赖已确认存在的 `tanh`。
这样这个算子不会因为"某个函数没导出"而在运行时才炸。
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tgai_go import train as T  # noqa: E402


def _tanh(x):
    """ORV 的 tanh **只在 Tensor 上是方法**（`x.tanh()`），没有模块级 `T.tanh`。

    实测：`T.tanh` 不存在（AttributeError），所以这里统一包一层，
    以后若补了模块级函数，只改这一个地方。
    """
    return x.tanh()


def _sigmoid(x):
    """sigmoid 用 tanh 恒等式实现，避免依赖未导出的算子。"""
    return 0.5 * (1.0 + _tanh(x * 0.5))


def situ_glu(x, w1, w2, w3, beta1=4.0, beta2=25.0):
    """SiTU-GLU 前向：`y = w3( situ(gate) * up )`，权重形状均为 (out, in)。"""
    g = x @ w1.transpose(0, 1)
    u = x @ w2.transpose(0, 1)
    gate = (_tanh(g * (1.0 / beta1)) * beta1) * _sigmoid(g)
    up = _tanh(u * (1.0 / beta2)) * beta2
    return (gate * up) @ w3.transpose(0, 1)


def situ_glu_numpy(x, w1, w2, w3, beta1=4.0, beta2=25.0):
    """同一公式的 numpy 版本，作为对拍基准（与 PyTorch 参考逐式对应）。

    sigmoid 用 `0.5·(1+tanh(x/2))` 恒等式：直接写 `1/(1+exp(-g))` 在 g 很大时会溢出报警
    （实测：CI 上 `-50·|W|` 量级的输入触发 `overflow encountered in exp`）。
    """
    g = x @ w1.T
    u = x @ w2.T
    gate = (beta1 * np.tanh(g / beta1)) * (0.5 * (1.0 + np.tanh(0.5 * g)))
    up = beta2 * np.tanh(u / beta2)
    return (gate * up) @ w3.T


# ═══════════════════════════════════════════════════════════
# V3 RoPE（支持 **partial rotary**）
# ═══════════════════════════════════════════════════════════

def rotate_half_perm(rot_dim):
    """构造"交错取反"的置换矩阵 `P`，使 `rotate_half(x) = x @ P`。

    参考实现（`model_v3.py::RotaryPositionEmbedding._rotate_half`）：
        (x1,x2,x3,x4) → (-x2,x1,-x4,x3)
    交错语义 ⇒ 偶数位取负的奇数位、奇数位取偶数位：
        out[2k]   = -x[2k+1]   ⇒ P[2k+1, 2k]   = -1
        out[2k+1] =  x[2k]     ⇒ P[2k,   2k+1] =  1

    用矩阵乘而不是原地切片赋值，因为 ORV 的张量不支持"对切片原地写"。
    """
    rd = int(rot_dim)
    if rd % 2:
        raise ValueError("rot_dim 必须是偶数，收到 %d。修法：partial 旋转的维数取偶数" % rd)
    P = np.zeros((rd, rd), dtype=np.float32)
    for k in range(rd // 2):
        P[2 * k + 1, 2 * k] = -1.0
        P[2 * k, 2 * k + 1] = 1.0
    return P


def rope_cos_sin(seq_len, rot_dim, theta, scaling=1.0):
    """生成 `(seq_len, rot_dim)` 的 cos/sin 表（与参考实现同式）。

    参考：`freqs = 1/(theta**(arange(0,rot_dim,2)/rot_dim))`，
    `emb = outer(pos, freqs).repeat_interleave(2, dim=-1)`。
    **注意分母是 rot_dim**（不是 dim），partial 旋转时这一点很关键。
    """
    rd = int(rot_dim)
    j = np.arange(0, rd, 2, dtype=np.float32) / float(rd)
    freqs = 1.0 / (float(theta) ** j)
    pos = np.arange(int(seq_len), dtype=np.float32) / float(scaling)
    emb = np.outer(pos, freqs).repeat(2, axis=1)          # (L, rd)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def v3_rope(x, cos, sin, permit=None):
    """对 `x`（形状 (..., rot_dim)）做交错 RoPE：`x*cos + rotate_half(x)*sin`。

    `permit` 是 `rotate_half_perm(x.shape[-1])`，可由调用方缓存复用。
    """
    rd = int(x.shape[-1])
    P = permit if permit is not None else rotate_half_perm(rd)
    P = T.tensor(np.ascontiguousarray(P, np.float32))
    return x * cos + (x @ P) * sin


def v3_rope_numpy(x, cos, sin):
    """同一公式的 numpy 版（交错 rotate_half），作为对拍基准。"""
    rd = x.shape[-1]
    rh = np.empty_like(x)
    rh[..., ::2] = -x[..., 1::2]
    rh[..., 1::2] = x[..., ::2]
    return x * cos + rh * sin


# ═══════════════════════════════════════════════════════════
# GlobalAttention（numpy 精确参考）
# ═══════════════════════════════════════════════════════════

def gqa_repeat(kv, n_q, n_kv):
    """把 KV 头按 `repeat_interleave` 复制到与 Q 头数一致。

    参考实现：`k.repeat_interleave(rep, dim=1)`，`rep = n_q // n_kv`
    —— 是**交错**复制（每头连续 rep 份），不是 tile。
    """
    if n_q == n_kv:
        return kv
    rep = int(n_q) // int(n_kv)
    return np.repeat(kv, rep, axis=1)


def attention_numpy(x, wq, wk, wv, wo, cos, sin, rot_dim, add_mask,
                    n_q, n_kv, rope_perm=None):
    """GlobalAttention 的 numpy 精确参考（训练路径，无 KV cache）。

    与 `model_v3.py::GlobalAttention.forward` 的对应关系：

        q,k,v   = 投影 → (B,n,T,d_k)
        q,k     = rope(仅前 rot_dim 维)
        k,v     = GQA 交错复制到 n_q 头
        out     = softmax(q·kᵀ/√d_k + mask) · v → 拼头 → o_proj

    `add_mask` 形状 `(B,1,T,S)` 或 `(T,S)`（加性，0 可见 / -inf 不可见）。
    """
    B, T, _ = x.shape
    d_k = wq.shape[0] // int(n_q)
    q = (x @ wq.T).reshape(B, T, int(n_q), d_k).transpose(0, 2, 1, 3)
    k = (x @ wk.T).reshape(B, T, int(n_kv), d_k).transpose(0, 2, 1, 3)
    v = (x @ wv.T).reshape(B, T, int(n_kv), d_k).transpose(0, 2, 1, 3)

    # RoPE 只作用在前 rot_dim 维（partial）
    def _rope(t):
        if rot_dim >= d_k:
            return v3_rope_numpy(t, cos[None, None], sin[None, None])
        head = v3_rope_numpy(t[..., :rot_dim], cos[None, None], sin[None, None])
        return np.concatenate([head, t[..., rot_dim:]], axis=-1)

    q = _rope(q)
    k = _rope(k)
    k = gqa_repeat(k, n_q, n_kv)
    v = gqa_repeat(v, n_q, n_kv)

    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(float(d_k))
    mask = add_mask if add_mask.ndim == 4 else add_mask[None, None]
    p = attention_numpy_softmax(scores, mask)
    out = p @ v                                   # (B,n,T,d_k)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, int(n_q) * d_k)
    return out @ wo.T


def attention_numpy_softmax(scores, add_mask):
    """数值稳定 softmax（掩码位置的权重严格为 0）。"""
    s = scores + add_mask
    m = np.max(s, axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(s - m)
    e = np.where(np.isfinite(add_mask), e, 0.0)
    den = e.sum(axis=-1, keepdims=True)
    return e / np.where(den > 0, den, 1.0)


# ═══════════════════════════════════════════════════════════
# MoEv3 路由（numpy 参考）
# ═══════════════════════════════════════════════════════════

def moe_route_numpy(x, router_w, expert_bias=None, top_k=2, balance="softmax"):
    """V3 MoE 的路由决策：返回 `(topi, gate, s_all)`。

    与参考实现（`model_v3.py::MoEv3.forward` ①–③）逐条对应：

        s_all = softmax(router(x), -1)          # 全专家 softmax（DeepSeek 口径）
        sel   = s_all + b                       # ★ b 只影响**选择**
        topi  = topk(sel, K)                    # 选择依据是 sel
        gate  = gather(s_all, topi)             # ★ 门控用**原始 s**，不是 sel

    `balance="quantile"`（K3 QB）时：`s = sigmoid(...)`，门控在选中集合内**重新归一化**。
    两种口径都实现，因为配置里 `moe_balance` 可切换。
    """
    W = np.asarray(router_w, np.float32)
    z = np.asarray(x, np.float32) @ W.T                      # (N, E)
    if balance == "quantile":
        s_all = 1.0 / (1.0 + np.exp(-z))
        sel = s_all if expert_bias is None else s_all + np.asarray(expert_bias, np.float32)
        topi = np.argsort(-sel, axis=-1, kind="stable")[:, :int(top_k)]
        gsel = np.take_along_axis(s_all, topi, axis=-1)
        den = np.maximum(gsel.sum(axis=-1, keepdims=True), 1e-9)
        gate = gsel / den
        return topi, gate, s_all

    m = z.max(axis=-1, keepdims=True)
    e = np.exp(z - m)
    s_all = e / e.sum(axis=-1, keepdims=True)
    sel = s_all if expert_bias is None else s_all + np.asarray(expert_bias, np.float32)
    # torch.topk 默认降序；这里用 stable 排序 + 次级键下标，保证可复现
    order = np.argsort(-sel, axis=-1, kind="stable")
    topi = order[:, :int(top_k)]
    gate = np.take_along_axis(s_all, topi, axis=-1)
    return topi, gate, s_all


def moe_update_bias(expert_bias, usage, gamma, top_k, n_experts, ema=None,
                    ema_decay=0.9):
    """偏置的**符号式**更新（参考实现 ④）：过载 `b -= γ`，欠载 `b += γ`。

    `usage`：本步每专家被选中的比例（长度 E，和为 `top_k`）。
    判定基准是"理想均匀负载"：每个专家应占 `top_k / E`。
    """
    b = np.array(expert_bias, np.float32, copy=True)
    u = np.asarray(usage, np.float32)
    ideal = float(top_k) / float(max(1, int(n_experts)))
    if ema is not None:
        u = ema_decay * np.asarray(ema, np.float32) + (1.0 - ema_decay) * u
    b += np.where(u < ideal, float(gamma), -float(gamma))
    return b


# ═══════════════════════════════════════════════════════════
# CSA（Compressed Sparse Attention）的两个关键步骤
# ═══════════════════════════════════════════════════════════

def csa_compress(h, c, seg=None):
    """序列压缩：把连续 `c` 个位置均值池化成 1 个 chunk，输出长度**恒为 ceil(T/c)**。

    参考实现（`model_v3.py::CompressedSparseAttention._compress`）：

    * `seg=None`：按固定 chunk 池化，**尾部复制补齐**；
    * `seg` 给出（每 token 的文档号）：**按文档边界分段池化**（每篇文档独立切 chunk、
      首尾相接），末尾**补零**到 `Tc`，无效 chunk 由掩码屏蔽。

    ★ 为什么必须按文档分段（P0-A 实测依据）：固定 chunk 会把两篇文档的 token 池进同一个
      chunk，于是文档 B 的 token **看不到自己文档开头那几个**（被一起屏蔽了）——
      权重看着对，但被池化的**内容本身脏了**。
    """
    x = np.asarray(h, np.float32)
    if x.ndim != 3:
        raise ValueError("h 形状应为 (B,T,L)，收到 %s。修法：把输入 reshape 成三维再压缩"
                         % (x.shape,))
    B, T, L = x.shape
    c = max(2, int(c))
    Tc = int(math.ceil(T / float(c)))
    out = np.zeros((B, Tc, L), np.float32)

    if seg is None:
        for b in range(B):
            for k in range(Tc):
                lo, hi = k * c, min((k + 1) * c, T)
                out[b, k] = x[b, lo:hi].mean(axis=0)
            if Tc * c > T:                       # 尾部复制补齐
                last = T - 1
                for k in range(Tc):
                    if k * c >= T:
                        out[b, k] = x[b, last]
        return out

    sd = np.asarray(seg)
    if sd.ndim == 1:
        sd = sd[None, :]
    for b in range(B):
        pos = 0
        doc_ids = []
        for t in range(T):
            if t == 0 or sd[b, t] != sd[b, t - 1]:
                doc_ids.append(t)
        doc_ids.append(T)
        chunks = []
        for di in range(len(doc_ids) - 1):
            lo, hi = doc_ids[di], doc_ids[di + 1]
            for k in range(lo, hi, c):
                chunks.append(x[b, k:min(k + c, hi)].mean(axis=0))
        for k, ch in enumerate(chunks[:Tc]):
            out[b, k] = ch
        # 不足 Tc 的 chunk 保持全零（无效 chunk 由掩码屏蔽）
    return out


def csa_conv_causal(x, w):
    """因果**深度可分离**卷积：`y[t] = Σ_{j<k} w[j] ⊙ x[t-j]`（越界补 0）。

    `x`：`(B,Tc,L)`；`w`：`(L, k)`（每通道一组核，depthwise）。
    ★ 只向左填充 ⇒ **不看未来**。参考实现记录过真 bug：旧代码用 `padding=k-1`（双向），
      再切前 Tc 个位置，导致每个位置仍能看见**未来 2 个 chunk**（实测 chunk 8 的输出
      在输入未变的情况下变化 1.450e+00）。
    """
    x = np.asarray(x, np.float32)
    B, Tc, L = x.shape
    w = np.asarray(w, np.float32)
    if w.shape[0] != L:
        raise ValueError("卷积核通道数 %d 与特征维 %d 不一致。修法：核应为 (L,k)"
                         % (w.shape[0], L))
    k = w.shape[1]
    pad = np.zeros((B, k - 1, L), np.float32)          # 只补左侧（历史）
    xp = np.concatenate([pad, x], axis=1)
    out = np.zeros((B, Tc, L), np.float32)
    for j in range(k):
        seg = xp[:, (k - 1 - j):(k - 1 - j + Tc), :]    # t-j
        out += seg * w[:, j][None, None, :]
    return out


# ═══════════════════════════════════════════════════════════
# GlobalAttention 的 ORV 张量实现
# ═══════════════════════════════════════════════════════════

def attention_orv(x, wq, wk, wv, wo, cos, sin, rot_dim, n_q, n_kv, causal=True,
                  attn_mask=None):

    """GlobalAttention 的 ORV 张量版（训练路径）。

    形状约定：`x` `(B,T,d_model)`；权重 `(out,in)`；`cos/sin` `(T, rot_dim)`。
    内部走 ORV 的 `scaled_dot_product_attention`，它接受 torch 风格的 `[B,H,S,Dh]`
    并自动展平（见 `train/__init__.py::_Functional.scaled_dot_product_attention`）。

    `attn_mask`：**加性**掩码，形状 `(B,1,T,T)`（0 可见 / -inf 不可见），
    由 `v3_masks.seg_additive` 造出来（已含因果）。给出时 `causal` 应传 False，
    否则因果会被叠加两次（结果仍等价，但多一次无用计算）。
    """
    B = int(x.shape[0])
    # ★ 不能把局部变量叫 `T`：那会遮蔽模块别名 `T`（= tgai_go.train），后面 `T.cat` 就会炸
    Tlen = int(x.shape[1])
    n_q, n_kv = int(n_q), int(n_kv)
    d_k = int(wq.shape[0]) // n_q

    q = (x @ wq.transpose(0, 1)).reshape(B, Tlen, n_q, d_k).transpose(1, 2)
    k = (x @ wk.transpose(0, 1)).reshape(B, Tlen, n_kv, d_k).transpose(1, 2)
    v = (x @ wv.transpose(0, 1)).reshape(B, Tlen, n_kv, d_k).transpose(1, 2)

    c = cos.reshape(1, 1, Tlen, int(rot_dim))
    s = sin.reshape(1, 1, Tlen, int(rot_dim))
    q = v3_rope(q, c, s)
    k = v3_rope(k, c, s)

    if n_kv != n_q:
        # GQA：**交错**复制（第 h 个 q 头用第 h//rep 个 kv 头）。
        # ORV 没有 expand/repeat_interleave，用 cat 逐头拼出来。
        rep = n_q // n_kv
        k = T.cat([k[:, h:h + 1] for h in range(n_kv) for _ in range(rep)], dim=1)
        v = T.cat([v[:, h:h + 1] for h in range(n_kv) for _ in range(rep)], dim=1)

    if attn_mask is not None:
        # SDPA 内部把 q 展平成 [B*H, S, S] 的分数，掩码也必须跟着变成同样的 batch 维：
        # (B,1,T,T) → (B,H,T,T) → (B*H,T,T)。ORV 没有 expand，用 cat 复制 H 份。
        am = attn_mask
        if am.shape[1] == 1 and n_q > 1:
            am = T.cat([am] * n_q, dim=1)
        am = am.reshape(B * n_q, int(am.shape[2]), int(am.shape[3]))
        out = T.functional.scaled_dot_product_attention(q, k, v, causal=causal,
                                                        attn_mask=am)
    else:
        out = T.functional.scaled_dot_product_attention(q, k, v, causal=causal)
    out = out.transpose(1, 2).reshape(B, Tlen, n_q * d_k)
    return out @ wo.transpose(0, 1)


def moe_route_orv(x, router_w, top_k, expert_bias=None):
    """MoEv3 路由的 ORV 张量版（**无偏置**路径）。

    与 `moe_route_numpy` 同口径：`s_all = softmax(router(x))`，`topk` 选专家，
    门控 = 选中的原始亲和度。

    ## 为什么只支持 `expert_bias=None`

    ORV 目前**没有 `gather`**（按任意下标取元素）。有偏置时：
        sel = s_all + b  → topi = topk(sel) → gate = s_all[topi]   ← 这一步需要 gather
    所以偏置版要等 gather 落地。训练从零开始时 `b = 0`，本路径覆盖起步阶段，
    也是"同权重同输入比 logits"够用的路径。
    """
    if expert_bias is not None:
        raise NotImplementedError(
            "带偏置的 MoE 路由暂不支持：ORV 没有 gather，无法在下标 topi 上取 s_all。"
            "修法：① 传 expert_bias=None（训练起步 b=0 时完全等价）；"
            "② 或先给 ORV 补一个按行 gather（运行时已有 gather_rows 可用于 embedding）；"
            "③ 或在 numpy 侧算好路由再灌回张量（仅离线对拍可用）")
    z = x @ router_w.transpose(0, 1)
    s_all = T.functional.softmax(z, axis=-1)
    topv, topi = T.topk(s_all, int(top_k), dim=-1)
    return topi, topv, s_all


def situ_glu_orv(x, w1, w2, w3, beta1=4.0, beta2=25.0):
    """SiTU-GLU 的 ORV 张量版（专家 FFN 的激活）。

    与 `situ_glu_numpy` 同式：

        gate = (β₁·tanh(g/β₁)) ⊙ sigmoid(g)     β₁=4
        up   = β₂·tanh(u/β₂)                    β₂=25
        y    = w3(gate ⊙ up)                    ⇒ 上界 β₁·β₂ = 100

    权重形状均为 `(out,in)`；`sigmoid` 仍用 `0.5·(1+tanh(x/2))` 恒等式（ORV 未导出 sigmoid）。
    """
    g = x @ w1.transpose(0, 1)
    u = x @ w2.transpose(0, 1)
    gate = (_tanh(g * (1.0 / beta1)) * beta1) * _sigmoid(g)
    up = _tanh(u * (1.0 / beta2)) * beta2
    return (gate * up) @ w3.transpose(0, 1)


def rmsnorm_orv(x, weight, eps=1e-6):
    """RMSNorm 的**函数式** ORV 版：`x / sqrt(mean(x^2) + eps) * w`。

    ORV 已有 `RMSNorm` 模块（CUDA 上走融合内核），但层组装需要一个不依赖模块实例的纯函数。
    参考实现用 `torch.rsqrt(mean(x^2) + eps)`，这里用 `sqrt` 后相除，数学等价。

    ★ 性能说明：这是**组合实现**（mean → add → sqrt → div → mul 五步），比融合内核多几次访存。
      先保证数值正确，速度留到优化阶段再换融合版 —— 这也是"下一步改什么"里的一条。
    """
    ms = (x * x).mean(axis=-1, keepdims=True)
    denom = (ms + float(eps)).sqrt()
    return (x / denom) * weight


def rmsnorm_numpy(x, weight, eps=1e-6):
    """同一公式的 numpy 版，作为对拍基准。"""
    x = np.asarray(x, np.float32)
    ms = (x * x).mean(axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)).astype(np.float32) * np.asarray(weight, np.float32)
