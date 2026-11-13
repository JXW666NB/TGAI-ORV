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
    """同一公式的 numpy 版本，作为对拍基准（与 PyTorch 参考逐式对应）。"""
    g = x @ w1.T
    u = x @ w2.T
    gate = (beta1 * np.tanh(g / beta1)) * (1.0 / (1.0 + np.exp(-g)))
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
