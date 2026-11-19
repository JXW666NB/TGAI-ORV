# -*- coding: utf-8 -*-
"""V3 的注意力掩码构造（纯 numpy，可离线验证）。

## 为什么单独抽出来

参考实现（`model_v3.py::GlobalAttention.forward`）里，掩码是全模块**最容易静默错**的一段：
打包训练时 `seg_mask` 是 `(B,1,T,T)` 的块对角（不同文档互不可见）。旧代码写成了

    if m.dim() == 4: m = m[:, 0]      # (B,1,T,T) → (B,T,T)
    ...
    attn_mask.masked_fill_(~m[0] ...)  # ← 只取 batch 0

于是 **batch 1..B-1 全都用 batch 0 的文档边界**：该隔开的没隔开（跨文档泄漏），
不该隔开的被隔开（文档内截断）。参考实现的注释记录了实测：B=2、batch0 分界=8 / batch1 分界=24 时，
与正确 4D 掩码最大差 **9.748e-01**，而与"只用 m[0]"的复刻只差 1.49e-07。

这里的函数**始终保留 batch 维**，并用测试钉住这个回归。

## 掩码约定

返回**加性掩码**（0 = 可见，-inf = 不可见），形状 `(B,1,T,S)`，可直接加到注意力分数上。
纯 numpy 是为了离线可测；上 GPU 时同一套语义换成 ORV 张量即可。
"""
import numpy as np

NEG_INF = float("-inf")


def causal_additive(T, S=None, sliding_window=None):
    """因果掩码：`(T, S)`；`S` 默认等于 `T`（训练时 S=T；增量解码时 S=klen>t）。"""
    T = int(T)
    S = int(T if S is None else S)
    qi = np.arange(T)[:, None]
    kj = np.arange(S)[None, :]
    # 允许看到的位置：kj <= qi + (S - T)（S>T 时右侧对齐到最新位置）
    visible = kj <= (qi + (S - T))
    if sliding_window:
        visible &= kj >= (qi - int(sliding_window) + 1)
    return np.where(visible, 0.0, NEG_INF).astype(np.float32)


def seg_additive(seg_mask, sliding_window=None):
    """把 bool 的 `seg_mask` 转成加性掩码，**保留 batch 维**。

    `seg_mask` 接受 `(B,T,S)`、`(B,1,T,S)` 或 `(T,S)`（单样本，广播到批）。
    返回 `(B,1,T,S)`。
    """
    m = np.asarray(seg_mask)
    if m.dtype != np.bool_:
        m = m.astype(np.bool_)
    if m.ndim == 4:
        m = m[:, 0]
    if m.ndim == 2:
        m = m[None, ...]
    if m.ndim != 3:
        raise ValueError("seg_mask 形状不支持：%s。修法：传 (B,T,S)/(B,1,T,S)/(T,S)"
                         % (m.shape,))
    B, T, S = m.shape
    if sliding_window:
        qi = np.arange(T)[:, None]
        kj = np.arange(S)[None, :]
        m = m & (kj >= (qi - int(sliding_window) + 1))[None, ...]
    out = np.where(m, 0.0, NEG_INF).astype(np.float32)
    return out[:, None, :, :]


def make_seg_mask(doc_ids):
    """按 token 的文档编号造**块对角因果** bool 掩码 `(B,1,T,T)`（打包训练用）。

    与参考实现（`data_v3.py::make_seg_mask`）**逐条对应**：

        mask[i, j] = (j <= i)                       # 因果（tril）
                     and (seg[j] == seg[i])         # 同一文档
                     and (seg[i] >= 0)              # pad 位置不可见

    ★ 因果必须**包含在这个掩码里**：模型侧调用 SDPA 时是 `is_causal=False`
      （见 `model_v3.py` 的 GlobalAttention），如果再叠一层因果就会与参考不一致。
    `doc_ids`：`(B,T)` 或 `(T,)` 的整数数组，同一文档的 token 编号相同；<0 视为 pad。
    """
    d = np.asarray(doc_ids)
    if d.ndim == 1:
        d = d[None, :]
    if d.ndim != 2:
        raise ValueError("doc_ids 形状不支持：%s。修法：传 (B,T) 或 (T,)" % (d.shape,))
    B, T = d.shape
    same = d[:, :, None] == d[:, None, :]                    # (B,T,T)
    causal = np.tril(np.ones((T, T), np.bool_))              # (T,T)
    valid = (d >= 0)[:, :, None]                             # (B,T,1)
    mask = same & causal[None, :, :] & valid
    return mask[:, None, :, :]


def attention_weights(scores, add_mask):
    """数值稳定的 softmax：`softmax(scores + mask)`，只用于离线验证（numpy）。"""
    s = np.asarray(scores, np.float32) + np.asarray(add_mask, np.float32)
    m = s.max(axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(s - m)
    e = np.where(np.isfinite(add_mask), e, 0.0)
    den = e.sum(axis=-1, keepdims=True)
    den = np.where(den > 0, den, 1.0)
    return e / den
