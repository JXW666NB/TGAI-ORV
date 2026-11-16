# -*- coding: utf-8 -*-
"""CSA（CompressedSparseAttention）的 ORV 实现 —— GPU 可跑 + autograd 可训练。

## 参考实现（**数学规范，最权威**）

`D:\\DSH-Workspace\\TGAI-对话档案\\_src15b_current\\model_v3.py` 的
`class CompressedSparseAttention`（第 566-961 行）。本文件逐式对应，不改数学。

## 结构（与参考实现一一对应）

```
kv = x @ wkv^T                                    # (B,T,latent)   低秩下投影
[压缩] 序列池化（每 c 个 token → 1 个 chunk）      # (B,Tc,latent)
       因果深度可分离卷积（kernel=k）              # (B,Tc,latent)
       RMSNorm(latent)                             # (B,Tc,latent)
k = hc @ wk^T, v = hc @ wv^T                       # (B,n_kv,Tc,dk)
q = x  @ wq^T                                      # (B,n_q,T,dk)
QK-Norm（对 d_k）→ 两个分支各自 RoPE
全局分支：q 看压缩后 chunk（因果、可块对角 seg 隔离）
局部分支：q 看最后 min(local_win,T) 段内的压缩 key（参考实现的掩码对齐）
out = out_g + sigmoid(gate) * out_l → 拼头 → @ wo^T
```

## ORV 上的实现约束（为什么这样写）

* **只用组合算子**：`@`(matmul) / `reshape` / `mean(axis)` / slice / `cat` /
  广播乘加 / `swap_last2` / `T.functional.softmax` / `nn_ops.rms_norm`。
  **不用** `einsum` / `topk` / `stack`（三者 CUDA 上抛 NotImplementedError，本项目踩过 3 次）。
* **不手写反向**：全部由组合算子自动建图；只有 `x[ids]`（取行）与切片这两种
  已有原语参与，反向由 ORV 自带 VJP 负责。
* **深度可分离卷积**没有 ORV 原语 ⇒ 用「左填充 + k 次平移相加」拼：
  `y[t] = Σ_j w[:,j] ⊙ x[t-j]`，k=3 时是 1 次 cat + 3 次切片 + 3 次乘加。
* **分段（seg）池化 / 分段卷积**需要「按文档边界搬数据」，而 ORV 没有
  `x[ids]=v`（不能回填）。因此全部改成**纯取行（gather）**形式：
  主机侧先算出「第 j 个成员 token 的平坦行号」，设备侧
  `hc = Σ_j x[idx_j] * mask_j / cnt`；分段卷积同理（源行号在主机侧算好，
  越界位置指向一个额外的全零行）。这样前/反向都能自动微分，且没有 Python 逐元素热路径。
* **软掩码用 -1e30 而不是 -inf**：ORV 的 softmax 遇到整行 -inf 会出 NaN；
  -1e30 在 fp32 下 exp 后严格下溢为 0，与 -inf 数值等价。整行都被掩掉的行
  （pad token / 局部分支的前 T-lk 行）用 `row_mask` 乘 0 —— 与 PyTorch SDPA
  对全掩码行返回 0 的语义**逐位一致**（已对拍，见下）。
* **注意力不走 ORV 的 SDPA，而是 4D 组合实现**（`(q @ kᵀ)*scale + mask → softmax → @v`）：
  本项目 SDPA 的 `attn_mask` 需要先展开成 `(B*H,T,S)`（每层多 H 倍显存），而 4D 广播
  只要 `(B,1,T,S)`。数学与 SDPA 完全等价（同一 scale、同一加性掩码）。
  代价：softmax 之前的 scores 会被物化（参考实现用 SDPA 的 mem-efficient 内核，
  显存更低）；这是**已知的显存差距**，见文末「已知限制」。

## 与 numpy 参考（`v3_ops.csa_compress` / `v3_ops.csa_conv_causal`）的关系

这两份 numpy 参考与 PyTorch 参考**并非处处相等**，实测差异（本文件已逐条钉住）：

| 位置 | numpy 参考 | PyTorch 参考（本实现照它） |
| --- | --- | --- |
| 固定 chunk 池化的尾部（`T % c != 0`） | 末尾 chunk 只对**实际存在的 token** 求均值 | **复制最后一个 token** 补齐到 c 再求均值（`T=10,c=4` ⇒ 末 chunk 为 9.75，numpy 为 9.5） |
| seg 分段池化 | 把 seg<0 的 pad token 也当成一篇“文档”参与切 chunk | 只对 valid token 切 chunk（pad 位置被丢弃） |
| seg 分段数 > `ceil(T/c)` 时 | 无处理 | **退化为固定切分**（`use_seg=False`） |

⇒ 本实现对拍时只在两者**数学等价**的域上比 numpy 参考（固定切分且 `T % c == 0`、
或 seg 分段且无 pad token），其余情况以 PyTorch 参考为准，并单独写断言钉住。

## 实测数字（2026-11-16，V100 / tgai-orv 0.9.83 / fp32）

* **前向**：ORV 张量 vs **PyTorch 参考**（`CompressedSparseAttention`，同权重同输入，
  经 `weights_from_torch_state_dict` 搬运）⇒ `max_abs = 2.384e-07`、`max_rel = 2.895e-07`
  （量级 0.8236；判据 1e-5，富余两个数量级）。脚本：`lab/csa_parity.py`。
* **反向**：`x_grad_max = 4.535e-01`，**10/10 个权重张量都拿到梯度**，
  输出形状 `(1,32,64)` 正确。脚本：`lab/csa_bwd.py`。
* numpy 参考（compress / conv）的对拍见 `tests/test_v3_csa.py`。

## 用法

```python
from tgai_go.train import v3_csa as C
cfg = C.CSAConfig(d_model=1024, n_heads=16, csa_compress=4)
w   = C.init_weights(np.random.default_rng(0), cfg)
cos, sin = C.rope_tables(cfg, seq_len=2048)      # 全表，可跨层/跨步复用
plan = C.make_plan(cfg, B=1, T=2048, seg=None)   # 与 seg 相关的掩码/索引，跨层复用
y = C.csa_forward(x, w, plan, cos, sin)         # (B,T,d_model)
```

## 已知限制（不藏）

1. 无 KV cache / 推理增量路径（本文件只做训练前向+反向）。
2. `head_max`（QK-Clip 遥测）未实现：本实现不返回参考实现的第二返回值。
3. 注意力 scores 物化 ⇒ 显存比 PyTorch 的 mem-efficient SDPA 高（待后续用
   ORV 的 lowmem 内核 + mask 融合解决，不属于本文件范围）。
4. 分段池化/卷积在 CUDA 上部分反向走 `x[ids]` 的主机中转（D2H+H2D），
   正确但非最优；单步开销实测见文末。
"""
import math

import numpy as np

import tgai_go.train as T
from tgai_go.train import nn_ops
from tgai_go.train import v3_ops as V

__all__ = [
    "CSAConfig", "CSAPlan", "make_plan", "rope_tables",
    "csa_forward", "csa_compress_orv", "csa_conv_causal_orv",
    "csa_conv_seg_orv", "init_weights", "weights_from_torch_state_dict",
]

# 掩码用的"极小值"（不是 -inf：ORV softmax 整行 -inf 会 NaN，原因见模块 docstring）
NEG_BIG = -1e30

# RoPE 的 rotate_half 置换矩阵缓存：{(rot_dim, device): Tensor}
_ROPE_PERM = {}


# ═════════════════════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════════════════════
class CSAConfig(object):
    """CSA 的配置（字段名与 `model_v3.py::TGAIV3Config` 一致，取值语义也对齐）。

    `csa_compress` / `csa_latent` / `csa_local_window` 的 0 值语义与参考实现完全相同：
    0 表示"用默认值"，默认值分别是 **无默认**（必须 >=2）、`max(64, d_model//4)`、
    `max(32, csa_compress*8)`。

    ★ 注意参考实现里 `latent` 的默认是 `max(64, d_model // 4)`（含 64 的下限），
    不是简单的 `d_model // 4` —— 小模型（d_model<256）时这一点会改变形状。
    """

    def __init__(self, d_model=1024, n_heads=16, n_kv_heads=0,
                 csa_compress=4, csa_latent=0, csa_conv_kernel=3,
                 csa_local_window=0, csa_seg_isolate=True, csa_causal_conv=True,
                 attn_qk_norm=True, rope_theta=500000.0, rope_partial=1.0,
                 max_seq_len=65536, norm_eps=1e-6):
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.csa_compress = max(2, int(csa_compress))
        self.csa_latent = int(csa_latent)
        self.csa_conv_kernel = int(csa_conv_kernel)
        self.csa_local_window = int(csa_local_window)
        self.csa_seg_isolate = bool(csa_seg_isolate)
        self.csa_causal_conv = bool(csa_causal_conv)
        self.attn_qk_norm = bool(attn_qk_norm)
        self.rope_theta = float(rope_theta)
        self.rope_partial = float(rope_partial)
        self.max_seq_len = int(max_seq_len)
        self.norm_eps = float(norm_eps)

    # ---- 与参考实现 TGAIV3Config 的属性逐条对应 ----
    @property
    def d_k(self):
        """每个注意力头的维度；n_heads 必须整除 d_model；奇数时减 1（与参考一致）。"""
        d = self.d_model // self.n_heads
        return d if d % 2 == 0 else d - 1

    @property
    def n_q_heads(self):
        return self.d_model // self.d_k

    @property
    def kv_heads(self):
        return self.n_kv_heads if self.n_kv_heads > 0 else self.n_q_heads

    @property
    def latent(self):
        return self.csa_latent if self.csa_latent > 0 else max(64, self.d_model // 4)

    @property
    def local_win(self):
        return (self.csa_local_window if self.csa_local_window > 0
                else max(32, self.csa_compress * 8))

    @property
    def rot_dim(self):
        """RoPE 实际旋转的维数（偶数；partial<=1）。"""
        rd = int(round(self.d_k * self.rope_partial))
        rd = rd if rd % 2 == 0 else rd - 1
        return max(2, min(rd, self.d_k))

    def conv_weight_shape(self):
        """ORV 侧的 depthwise 卷积核形状 (latent, k) —— **不是** PyTorch 的 (C,1,k)。"""
        return (self.latent, self.csa_conv_kernel)

    def __repr__(self):
        return ("CSAConfig(d_model=%d, n_heads=%d, n_kv=%d, d_k=%d, latent=%d, "
                "c=%d, k=%d, local_win=%d, seg_isolate=%s, causal_conv=%s, "
                "qk_norm=%s, rot_dim=%d)"
                % (self.d_model, self.n_heads, self.kv_heads, self.d_k, self.latent,
                   self.csa_compress, self.csa_conv_kernel, self.local_win,
                   self.csa_seg_isolate, self.csa_causal_conv,
                   self.attn_qk_norm, self.rot_dim))


def cos_sin_tables(cfg, seq_len=None):
    """生成 (seq_len, rot_dim) 的 cos/sin numpy 表（与参考实现的 RoPE 同式）。

    参考实现：`freqs = 1/θ**(arange(0,rot_dim,2)/rot_dim)`，
    `emb = outer(pos, freqs).repeat_interleave(2, -1)`；pos = arange(L)/scaling。
    """
    n = int(seq_len if seq_len is not None else cfg.max_seq_len)
    return V.rope_cos_sin(n, cfg.rot_dim, cfg.rope_theta)


def rope_tables(cfg, seq_len=None):
    """同 `cos_sin_tables`，但直接返回 ORV 张量（放在当前默认设备）。"""
    cos, sin = cos_sin_tables(cfg, seq_len)
    return T.tensor(cos), T.tensor(sin)


# ═════════════════════════════════════════════════════════════════════════
# 小工具：RoPE / rmsnorm / 张量整理
# ═════════════════════════════════════════════════════════════════════════
def _rope_perm(rot_dim, device):
    """`rotate_half` 的交错置换矩阵（缓存；参考实现 `_rotate_half` 逐位对应）。"""
    key = (int(rot_dim), str(device))
    t = _ROPE_PERM.get(key)
    if t is None:
        p = V.rotate_half_perm(int(rot_dim))          # (rd, rd) 交错取反的置换矩阵
        t = T.tensor(np.ascontiguousarray(p, np.float32), device=device)
        if len(_ROPE_PERM) > 16:
            _ROPE_PERM.clear()
        _ROPE_PERM[key] = t
    return t


def _rope(x, cos, sin, device):
    """交错 RoPE（支持 partial）：`x*cos + rotate_half(x)*sin`。

    `cos/sin`：(1,1,L,rot_dim)；`x`：(B,H,L,dk)。partial 时只转前 rot_dim 维，
    其余原样通过（参考实现 `RotaryPositionEmbedding.forward` 的 partial 分支）。
    """
    rd = int(cos.shape[-1])
    perm = _rope_perm(rd, device)
    if rd >= int(x.shape[-1]):
        return x * cos + (x @ perm) * sin
    xr = x[..., :rd]
    xp = x[..., rd:]
    return T.cat([xr * cos + (xr @ perm) * sin, xp], dim=-1)


def _rms(x, w, eps):
    """RMSNorm（组合式，与参考实现的公式逐式一致）。x 的最后一维 = len(w)。"""
    return nn_ops.rms_norm(x, w, eps)


def _heads(x, n_heads, d_k):
    """(B,L,nh*dk) → (B,nh,L,dk)。"""
    B = int(x.shape[0])
    L = int(x.shape[1])
    return x.reshape(B, L, int(n_heads), int(d_k)).transpose(1, 2)


def _zeros(shape, device):
    return T.zeros(tuple(int(s) for s in shape), device=device)


def _ones(shape, device):
    return T.ones(tuple(int(s) for s in shape), device=device)


# ═════════════════════════════════════════════════════════════════════════
# 主机侧：chunk 划分（严格照参考实现 forward 里的 cid / n_chunk 规则）
# ═════════════════════════════════════════════════════════════════════════
def _chunk_layout(seg, c):
    """按参考实现规则把每个 valid token 分配到 chunk，返回主机侧索引。

    参考实现（`model_v3.py:815-828`）：
        bnd[b, t] = (t % c == 0) or (seg[b,t] != seg[b,t-1])；bnd[:,0] = True
        cid_raw   = cumsum(bnd) - 1
        cid       = where(valid, cid_raw, 0)
        n_chunk[b]= max(where(valid, cid_raw+1, 0))

    返回 dict（全部 numpy，chunk 槽位按 `b*Tc + cid` 展平）：
        valid (B,T) bool / cid (B,T) int64 / n_chunk (B,)
        cnt   (B*Tc,)         每个 chunk 的 valid token 数
        offs  (B*Tc,)         该 chunk 在 vpos 里的起始偏移
        slot_doc (B*Tc,)      该 chunk 的文档标签（-1 = 空 chunk）
        slot_pos (B*Tc,)      该 chunk 在该文档内的序号 p（0 起）
        vpos  (n_valid,)      valid token 的平坦下标（b*T+t），升序
        doc_ord (B,T)         每个 valid token 的文档序号（按出现顺序，1 起；其余 0）
    """
    seg = np.asarray(seg, np.int32)
    if seg.ndim == 1:
        seg = seg[None, :]
    if seg.ndim != 2:
        raise ValueError("seg 形状应为 (B,T)，收到 %s。修法：把文档号放进二维数组"
                         % (seg.shape,))
    B, Tlen = seg.shape
    c = int(c)
    Tc = int(math.ceil(Tlen / float(c)))
    valid = seg >= 0

    bnd = np.zeros((B, Tlen), np.int64)
    bnd[:, 0] = 1
    if Tlen > 1:
        tt = np.arange(Tlen)[None, :]
        bnd[:, 1:] = (((tt[:, 1:] % c) == 0)
                      | (seg[:, 1:] != seg[:, :-1])).astype(np.int64)
    cid_raw = np.cumsum(bnd, axis=1) - 1
    cid = np.where(valid, cid_raw, 0)
    n_chunk = np.where(valid, cid_raw + 1, 0).max(axis=1).astype(np.int64)

    # ★ 2026-11-16 修（真实分片上实测崩过）：分段模式会额外开 chunk。
    #   文档边界处的 `seg[b,t] != seg[b,t-1]` 也会置 bnd=1，所以「总 chunk 数」
    #   = Σ_docs ceil(doc_len/c)，**可以远大于** ceil(T/c)（一篇 1 token 的文档
    #   就占一个 chunk）。若仍按 ceil(T/c) 分配 `B*Tc` 个槽位，越界的 slot 会让
    #   `np.bincount(..., minlength=B*Tc)` 返回更长的数组，随后
    #   `slot_doc`（长 B*Tc+1）与 `new_doc`（长 B*Tc）长度不一致，
    #   报 "could not broadcast input array from shape (128,) into shape (127,)"。
    #   修法：Tc 取「数学下界」与「实际用到的 chunk 数」的较大者。
    if n_chunk.size:
        Tc = int(max(Tc, int(n_chunk.max())))

    # valid token 的平坦下标（升序）与「每篇文档的序号」
    vpos = np.nonzero(valid.reshape(-1))[0].astype(np.int64)
    doc_start = np.zeros((B, Tlen), bool)
    doc_start[:, 0] = True
    if Tlen > 1:
        doc_start[:, 1:] = seg[:, 1:] != seg[:, :-1]
    doc_ord = np.cumsum(doc_start.astype(np.int64), axis=1)
    doc_ord = np.where(valid, doc_ord, 0)

    cnt = np.zeros(B * Tc, np.int64)
    slot_doc = np.full(B * Tc, -1, np.int64)
    slot_pos = np.zeros(B * Tc, np.int64)
    offs = np.zeros(B * Tc, np.int64)
    vpos = vpos.astype(np.int64)
    if vpos.size:
        b_of = vpos // Tlen
        c_of = cid.reshape(-1)[vpos]
        slot = b_of * Tc + c_of
        # slot 随 vpos 单调不减 ⇒ bincount 后做前缀和即得到每 chunk 的起始偏移
        cnt = np.bincount(slot, minlength=B * Tc).astype(np.int64)
        cs = np.cumsum(cnt)
        offs = np.concatenate([[0], cs[:-1]]).astype(np.int64)
        # 每个有成员的 chunk：文档标签 / 文档内序号
        has = cnt > 0
        first = np.minimum(offs, max(vpos.size - 1, 0))
        first_tok = vpos[first]
        slot_doc = np.where(has, doc_ord.reshape(-1)[first_tok], -1)
        # 文档内序号 p = 当前 chunk 与该文档首个 chunk 之差
        new_doc = np.zeros(B * Tc, bool)
        new_doc[0] = True
        new_doc[1:] = slot_doc[1:] != slot_doc[:-1]
        first_slot = np.maximum.accumulate(np.where(new_doc, np.arange(B * Tc), 0))
        slot_pos = np.arange(B * Tc) - first_slot
        slot_pos = np.where(has, slot_pos, 0)
    # 池化的取行计划：第 j 个成员 token 的平坦行号 + 有效掩码 + 1/cnt
    pool_idx, pool_mask = [], []
    nslot = B * Tc
    for j in range(c):
        has = cnt > j
        if vpos.size:
            src = np.minimum(np.where(has, offs + j, 0), vpos.size - 1)
            idx_j = np.where(has, vpos[src], vpos[0] if vpos.size else 0)
        else:
            idx_j = np.zeros(nslot, np.int64)
        pool_idx.append(idx_j.astype(np.int64))
        pool_mask.append(np.ascontiguousarray(has.astype(np.float32).reshape(nslot, 1)))
    pool_inv = np.ascontiguousarray(
        np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 0.0).astype(np.float32).reshape(nslot, 1))
    out = dict(valid=valid, cid=cid, n_chunk=n_chunk, cnt=cnt, offs=offs,
               slot_doc=slot_doc, slot_pos=slot_pos, vpos=vpos,
               doc_ord=doc_ord, Tc=Tc, B=B, Tlen=Tlen,
               pool_idx=pool_idx, pool_mask=pool_mask, pool_inv=pool_inv,
               all_valid=bool(cnt.all()))
    return out


def _pool_from_layout(h, c, layout, Tc):
    """把 (B,T,L) 按 chunk 分段池化 → (B,Tc,L)（纯取行 + 掩码乘加，无回填）。

    chunk k 的成员 token 是 vpos[offs[k] .. offs[k]+cnt[k]-1]（顺序即 token 顺序），
    所以第 j 个成员就是 vpos[offs[j 号槽位] + j]；越界槽位指向一个合法行并用 0 掩掉。
    取行计划在 `_chunk_layout` 里算好（plan 复用），这里只做设备侧乘加。
    """
    B, Tlen, L = int(h.shape[0]), int(h.shape[1]), int(h.shape[2])
    c = int(c)
    Tc = int(Tc)
    device = h.device
    h_flat = h.reshape(B * Tlen, L)                  # 取行原语只支持首轴一维索引

    acc = None
    for j in range(c):
        got = h_flat[layout["pool_idx"][j]]          # (nslot, L)；numpy 索引数组
        if not layout["all_valid"]:
            got = got * T.tensor(layout["pool_mask"][j], device=device)
        acc = got if acc is None else acc + got
    acc = acc * T.tensor(layout["pool_inv"], device=device)
    return acc.reshape(B, Tc, L)


def _conv_seg_idx(layout, k):
    """分段卷积的取行计划：第 j 个抽头的源行号（展平到 B*(Tc+1) 行空间）。

    每篇文档单独因果卷积（参考实现 `_compress` 的逐段 conv）：
        out[p] = Σ_j w[j] · hc_doc[p-j]      （p-j < 0 的位置贡献 0）
    第 `B*(Tc+1)-1` 行是额外的一行全零，充当"越界 / 空 chunk"的零源。
    """
    B, Tc = layout["B"], layout["Tc"]
    nslot = B * Tc
    row = np.arange(nslot)
    b_of = row // Tc
    k_of = row % Tc
    cnt = layout["cnt"]
    pos = layout["slot_pos"]
    dummy = b_of * (Tc + 1) + Tc
    out = []
    for j in range(int(k)):
        ok = (cnt > 0) & (pos >= j)
        src = np.where(ok, b_of * (Tc + 1) + (k_of - j), dummy).astype(np.int64)
        out.append(np.minimum(np.maximum(src, 0), B * (Tc + 1) - 1))
    return out


# ═════════════════════════════════════════════════════════════════════════
# 主机侧：掩码（严格照参考实现 forward 里的 cmask / lmask 规则）
# ═════════════════════════════════════════════════════════════════════════
def _mask_global_fixed(Tlen, Tc, c, B):
    """无 seg 的压缩分支掩码：token i 可见 chunk k ⇔ k <= i // c。"""
    iq = np.arange(Tlen) // int(c)
    ik = np.arange(Tc)
    vis = (ik[None, :] <= iq[:, None])                       # (T,Tc)
    m = np.where(vis, 0.0, NEG_BIG).astype(np.float32)
    return np.ascontiguousarray(np.broadcast_to(m, (B, 1, Tlen, Tc)))


def _mask_local_fixed(Tlen, lk, B):
    """局部分支掩码：与参考实现同式的**右对齐因果**（`triu(diagonal=lk-lq+1)`）。

    ★ 这不是"每个 token 看自己的局部窗口"：k 是**压缩后**的 chunk（长 lk），
    q 是 T 个 token，参考实现的掩码把两者按"右对齐"比较 ⇒ 只有最后 lk 个 token
    能看见 key（前 T-lk 行整行被掩，输出 0）。本文件照抄该语义（已与 PyTorch 对拍）。
    """
    iq = np.arange(Tlen)[:, None]
    ik = np.arange(lk)[None, :]
    vis = ik < (iq + (lk - Tlen + 1))
    m = np.where(vis, 0.0, NEG_BIG).astype(np.float32)
    row = vis.any(axis=1).astype(np.float32)
    return (np.ascontiguousarray(np.broadcast_to(m, (B, 1, Tlen, lk))),
            np.ascontiguousarray(np.broadcast_to(row.reshape(1, 1, Tlen, 1), (B, 1, Tlen, 1))))


def _mask_global_seg(seg, layout, Tc):
    """seg 路径的压缩分支掩码（块对角 + 因果 + 只对有效 chunk）。

    参考实现（`model_v3.py:869-903`）：
        cseg[b,k]      = chunk k 所属文档号（空 chunk = -1）
        valid_chunk    = 该 chunk 被至少一个 valid token 占用
        tok_chunk[b,i] = token i 落在哪个 chunk
        cmask[b,i,k]   = (cseg[b,k] == seg[b,i]) & valid_chunk[b,k] & (k <= tok_chunk[b,i])
    """
    seg = np.asarray(seg, np.int32)
    B, Tlen = seg.shape
    nslot = B * Tc
    cnt = layout["cnt"]
    slot_doc = layout["slot_doc"]
    cid = layout["cid"]
    cseg = slot_doc.reshape(B, Tc)
    valid_chunk = (cnt > 0).reshape(B, Tc) & (cseg >= 0)
    tok_chunk = cid
    ar = np.arange(Tc)[None, None, :]
    cmask = ((cseg.reshape(B, 1, Tc) == seg.reshape(B, Tlen, 1))
             & valid_chunk.reshape(B, 1, Tc)
             & (ar <= tok_chunk.reshape(B, Tlen, 1)))
    m = np.where(cmask, 0.0, NEG_BIG).astype(np.float32)
    row = cmask.any(axis=-1).astype(np.float32)
    return (np.ascontiguousarray(m.reshape(B, 1, Tlen, Tc)),
            np.ascontiguousarray(row.reshape(B, 1, Tlen, 1)))


def _mask_local_seg(seg, layout, Tc, lk):
    """seg 路径的局部分支掩码（与参考实现 `model_v3.py:929-943` 同式）。

    key 是压缩后的 chunk（最后 lk 个），可见条件 = 同文档 & 该 chunk 有效 & 因果
    （`key_chunk <= tok_chunk`）& 参考实现的右对齐比较 `j <= i - (lq-lk)`。
    两者同时成立才可见。
    """
    seg = np.asarray(seg, np.int32)
    B, Tlen = seg.shape
    nslot = B * Tc
    cnt = layout["cnt"]
    slot_doc = layout["slot_doc"]
    cid = layout["cid"]
    cseg = slot_doc.reshape(B, Tc)
    valid_chunk = (cnt > 0).reshape(B, Tc) & (cseg >= 0)
    tok_chunk = cid
    cl = cseg[:, Tc - lk:]
    cl_ok = valid_chunk[:, Tc - lk:]
    key_chunk = np.arange(Tc - lk, Tc)[None, None, :]
    iq = np.arange(Tlen)[None, :, None]
    right_aligned = np.arange(lk)[None, None, :] < (iq + (lk - Tlen + 1))
    same = ((seg.reshape(B, Tlen, 1) == cl.reshape(B, 1, lk))
            & (seg >= 0).reshape(B, Tlen, 1)
            & cl_ok.reshape(B, 1, lk))
    vis = same & (key_chunk <= tok_chunk.reshape(B, Tlen, 1)) & right_aligned
    m = np.where(vis, 0.0, NEG_BIG).astype(np.float32)
    row = vis.any(axis=-1).astype(np.float32)
    return (np.ascontiguousarray(m.reshape(B, 1, Tlen, lk)),
            np.ascontiguousarray(row.reshape(B, 1, Tlen, 1)))


# ═════════════════════════════════════════════════════════════════════════
# Plan：与 seg 相关的一切（索引/掩码/切片）只在主机侧算一次，跨层复用
# ═════════════════════════════════════════════════════════════════════════
class CSAPlan(object):
    """一个 batch 的 CSA 预计算计划。**跨层复用**（同一 forward 里 12 个 CSA 层共用）。

    字段全部是 numpy（`_n` 前缀）或张量（`_t` 前缀）。张量按设备懒构建并缓存，
    所以 plan 可以在 CPU 上建、随后被 CUDA 前向使用（反之亦然）。

    关键语义（与参考实现 `CompressedSparseAttention.forward` 完全一致）：
      * `use_seg=False`：seg 未给 / `csa_seg_isolate=False` / 分段数超限退化的固定切分；
      * `single=True`：每样本最多 1 篇文档 —— 池化仍按文档边界（等价固定网格），
        但卷积走整段快路径（参考实现的 `single` 分支）；
      * `fallback=True`：分段后 chunk 数 > `ceil(T/c)` ⇒ 参考实现**退化回固定切分**
        （这时连掩码也一起退回固定因果；不这样做会把两篇文档的 chunk 合并、破坏隔离）。
    """

    __slots__ = ("cfg", "B", "Tlen", "Tc", "c", "seg", "use_seg", "single",
                 "fallback", "w", "lk", "off_local", "_n", "_t", "_rope")

    def __init__(self, cfg, B, Tlen, seg=None):
        self.cfg = cfg
        self.B = int(B)
        self.Tlen = int(Tlen)
        self.c = int(cfg.csa_compress)
        self.Tc = int(math.ceil(self.Tlen / float(self.c)))
        self.seg = None if seg is None else np.asarray(seg, np.int32)
        self.fallback = False
        self.single = False
        self.use_seg = False
        self._t = {}
        self._rope = {}
        self._n = {}

        layout = None
        if self.seg is not None and cfg.csa_seg_isolate:
            if self.seg.shape != (self.B, self.Tlen):
                raise ValueError(
                    "seg 形状应为 (B,T)=(%d,%d)，收到 %s。修法：seg 是每个 token 的文档号，"
                    "<0 表示 padding" % (self.B, self.Tlen, (self.seg.shape,)))
            layout = _chunk_layout(self.seg, self.c)
            if int(layout["n_chunk"].max() if layout["n_chunk"].size else 0) > self.Tc:
                # 参考实现同款退化：分段 chunk 数超过 ceil(T/c) ⇒ 回固定切分
                self.fallback = True
                self.seg = None
                layout = None
            else:
                self.use_seg = True
                self.single = bool(int(layout["n_chunk"].max()) <= 1)

        self.w = min(int(cfg.local_win), self.Tlen)
        self.lk = min(self.w, self.Tc)
        self.off_local = max(0, self.Tlen - self.w)

        n = self._n
        if self.use_seg and not self.single:
            n["conv_idx"] = _conv_seg_idx(layout, cfg.csa_conv_kernel)
        if self.use_seg:
            n["pool"] = dict(layout=layout)
        else:
            n["pad"] = (self.c - self.Tlen % self.c) % self.c

        if self.use_seg:
            mg, rg = _mask_global_seg(self.seg, layout, self.Tc)
            ml, rl = _mask_local_seg(self.seg, layout, self.Tc, self.lk)
        else:
            mg = _mask_global_fixed(self.Tlen, self.Tc, self.c, self.B)
            rg = None
            ml, rl = _mask_local_fixed(self.Tlen, self.lk, self.B)
        n["mask_g"], n["row_g"], n["mask_l"], n["row_l"] = mg, rg, ml, rl

    # ---- 张量懒构建（按设备缓存；索引数组留 numpy，取行原语直接吃 numpy 索引） ----
    def _tensors(self, device):
        device = str(device)
        d = self._t.get(device)
        if d is None:
            d = {}
            for key in ("mask_g", "row_g", "mask_l", "row_l"):
                arr = self._n.get(key)
                if arr is not None:
                    d[key] = T.tensor(np.ascontiguousarray(arr), device=device)
            self._t[device] = d
        return d

    def _rope_views(self, cos, sin):
        """切出本 plan 需要的三组 cos/sin（q / 全局分支的 k chunk / 局部分支的 k）。

        ★ 三组位置切片都来自同一张全表（参考实现 g_rope / l_rope 的 θ 与 partial 相同）：
          · q 与局部 q：位置 0..T-1
          · 全局 k：位置 0..Tc-1（参考实现 k_g = g_rope(k)，offset=0）
          · 局部 k：位置 offset..offset+lk-1，offset = max(0, T-w)
        缓存键带上 cos/sin 的 id 并持有引用（防止 id 复用）。
        """
        key = (id(cos), id(sin), str(cos.device))
        got = self._rope.get(key)
        if got is not None:
            return got
        rd = int(cos.shape[-1])
        def _view(table, start, length):
            a = table[start:start + length]                     # (L, rd)
            return a.reshape(1, 1, int(length), rd)

        cos_q = _view(cos, 0, self.Tlen)
        sin_q = _view(sin, 0, self.Tlen)
        cos_k = _view(cos, 0, self.Tc)
        sin_k = _view(sin, 0, self.Tc)
        cos_kl = _view(cos, self.off_local, self.lk)
        sin_kl = _view(sin, self.off_local, self.lk)
        out = dict(cos_q=cos_q, sin_q=sin_q, cos_k=cos_k, sin_k=sin_k,
                   cos_kl=cos_kl, sin_kl=sin_kl, _hold=(cos, sin))
        if len(self._rope) > 8:
            self._rope.clear()
        self._rope[key] = out
        return out


# 全局 plan 缓存：训练里 seg 每步变，但 seg=None 时 plan 是常量（省掉每步重建）。
_PLAN_CACHE = {}


def make_plan(cfg, B, Tlen, seg=None, cache=True):
    """构造 `CSAPlan`。

    `seg`：(B,T) 文档号（numpy/int，<0 = padding）或 None。
    `cache=True` 时对**内容相同**的 (cfg, B, T, seg) 复用 plan（seg=None 时跨步复用；
    seg 变化时自动新建）。缓存只放主机侧 numpy，不占显存。
    """
    key = None
    if cache:
        sig = None
        if seg is not None:
            s = np.ascontiguousarray(np.asarray(seg, np.int32))
            sig = (s.shape, hash(s.tobytes()))
        key = (id(cfg), int(B), int(Tlen), id(cfg) if cfg is not None else 0,
               cfg.csa_compress, cfg.local_win, cfg.csa_seg_isolate,
               cfg.csa_causal_conv, cfg.csa_conv_kernel, sig)
        got = _PLAN_CACHE.get(key)
        if got is not None:
            return got
    plan = CSAPlan(cfg, B, Tlen, seg)
    if key is not None:
        plan._n["_cfg_hold"] = cfg           # 防止 id(cfg) 被复用
        if seg is not None:
            # ★ 2026-11-17：`seg` 随 batch 变化 ⇒ 这条 plan **下一步不会再命中**，
            #   但 plan 会在 GPU 上懒建并缓存张量（`_t`），攒到 16 条就白占显存。
            #   实测 1.5B / seq1024 只差几十 MB 装不下，正是这类"用不到的缓存"。
            #   这里只保留**最近一条**按 seg 建的 plan（同一 forward 内 18 个 CSA 层
            #   共用同一份，命中不受影响），静态 plan（seg is None）仍然留在字典里跨步复用。
            for _k in [k for k in _PLAN_CACHE if k[-1] is not None]:
                _PLAN_CACHE.pop(_k, None)
        elif len(_PLAN_CACHE) > 16:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[key] = plan
    return plan


# ═════════════════════════════════════════════════════════════════════════
# 两个"原语"：与 v3_ops 的 numpy 参考对应
# ═════════════════════════════════════════════════════════════════════════
def csa_compress_orv(h, c, seg=None):
    """序列压缩池化（张量版），对应 `v3_ops.csa_compress` + 参考实现 `_compress` 的池化部分。

    * `seg=None`：固定 chunk 均值池化；`T % c != 0` 时**复制最后一个 token**补齐
      （参考实现口径；numpy 参考在这里是"只对实际 token 求均值"，两者只在 `T % c == 0` 时相等）。
    * `seg` 给出：(B,T) 文档号；按文档边界切 chunk，只对 valid token 池化，
      空 chunk 填 0。分段 chunk 数 > ceil(T/c) 时**截断**（参考实现 `_compress` 的 `acc[:Tc]`）。
    """
    h = h if isinstance(h, T.Tensor) else T.tensor(np.ascontiguousarray(h, np.float32))
    c = max(2, int(c))
    B, Tlen, L = int(h.shape[0]), int(h.shape[1]), int(h.shape[2])
    Tc = int(math.ceil(Tlen / float(c)))
    if seg is None:
        pad = (c - Tlen % c) % c
        if pad:
            last = h[:, Tlen - 1:Tlen, :]                 # (B,1,L)
            h = T.cat([h] + [last] * pad, dim=1)
        return h.reshape(B, Tc, c, L).mean(axis=2)

    layout = _chunk_layout(seg, c)
    if layout["Tlen"] != Tlen or layout["B"] != B:
        raise ValueError("seg 形状与 h 不一致：h=%s seg=%s。修法：seg 应为 (B,T)"
                         % ((B, Tlen), (layout["B"], layout["Tlen"])))
    return _pool_from_layout(h, c, layout, Tc)


def csa_conv_causal_orv(x, w):
    """因果深度可分离卷积（整段，张量版），对应 `v3_ops.csa_conv_causal`。

    `x`：(B,Tc,L)；`w`：(L,k)，语义与 numpy 参考一致：`y[t] = Σ_j w[:,j] ⊙ x[t-j]`。

    ★ 与 PyTorch `nn.Conv1d` 的核**方向相反**：参考实现是 `nn.Conv1d(padding=左 k-1)`
    + 切前 n_out 个位置，等价于 `y[t] = Σ_j w_torch[j] · x[t-(k-1)+j]`，
    即 `w_orv[j] = w_torch[k-1-j]`（见 `weights_from_torch_state_dict`）。
    """
    x = x if isinstance(x, T.Tensor) else T.tensor(np.ascontiguousarray(x, np.float32))
    w = w if isinstance(w, T.Tensor) else T.tensor(np.ascontiguousarray(w, np.float32))
    B, Tc, L = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    k = int(w.shape[1])
    if int(w.shape[0]) != L:
        raise ValueError("卷积核通道数 %d 与特征维 %d 不一致。修法：核应为 (L,k)"
                         % (int(w.shape[0]), L))
    if k <= 1:
        return x * w[:, 0]
    xp = T.cat([_zeros((B, k - 1, L), x.device), x], dim=1)     # 只补左侧（因果）
    out = xp[:, k - 1:k - 1 + Tc, :] * w[:, 0]
    for j in range(1, k):
        out = out + xp[:, k - 1 - j:k - 1 - j + Tc, :] * w[:, j]
    return out


def csa_conv_seg_orv(x, w, plan):
    """分段因果深度可分离卷积（每篇文档单独卷积，张量版）。

    `x`：(B,Tc,L) 已按文档分段池化的结果；`w`：(L,k)（ORV 方向，见上）。
    用 plan 里预先算好的源行号做 k 次取行：源行落在同一篇文档内，
    越界/空 chunk 指向末行全零 ⇒ 与参考实现的逐段 conv 数值一致。
    """
    x = x if isinstance(x, T.Tensor) else T.tensor(np.ascontiguousarray(x, np.float32))
    w = w if isinstance(w, T.Tensor) else T.tensor(np.ascontiguousarray(w, np.float32))
    B, Tc, L = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    k = int(w.shape[1])
    idxs = plan._n.get("conv_idx")
    if idxs is None:
        raise ValueError(
            "plan 里没有分段卷积计划（单文档/无 seg 时应走整段卷积）。修法："
            "多文档 seg 用 make_plan(..., seg=...) 并检查 plan.single")
    x_ext = T.cat([x.reshape(B, Tc, L), _zeros((B, 1, L), x.device)], dim=1)
    x_ext = x_ext.reshape(B * (Tc + 1), L)          # 每篇样本末尾多一行全零（越界源）
    out = x_ext[idxs[0]] * w[:, 0]
    for j in range(1, k):
        out = out + x_ext[idxs[j]] * w[:, j]
    return out.reshape(B, Tc, L)


def _conv_whole(x, w, causal=True):
    """整段卷积；`causal=False` 时按参考实现的 A/B 回退做**双向**填充。

    ★ 数学注记（已实测）：参考实现的"双向填充 + 切前 n_out"与"只左填充"在
    `y[t] = Σ_j w_torch[j] x[t-(k-1)+j]` 上**完全等价**（右侧多出来的零不参与
    前 n_out 个输出）。这里保留两种写法只是为了逐符号对应那个开关。
    """
    if causal:
        return csa_conv_causal_orv(x, w)
    x = x if isinstance(x, T.Tensor) else T.tensor(np.ascontiguousarray(x, np.float32))
    w = w if isinstance(w, T.Tensor) else T.tensor(np.ascontiguousarray(w, np.float32))
    B, Tc, L = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    k = int(w.shape[1])
    if k <= 1:
        return x * w[:, 0]
    xp = T.cat([_zeros((B, k - 1, L), x.device), x, _zeros((B, k - 1, L), x.device)], dim=1)
    out = xp[:, k - 1:k - 1 + Tc, :] * w[:, 0]
    for j in range(1, k):
        out = out + xp[:, k - 1 - j:k - 1 - j + Tc, :] * w[:, j]
    return out


# ═════════════════════════════════════════════════════════════════════════
# 注意力（4D 组合实现；数学 = SDPA(scale=1/sqrt(d_k), 加性 mask)）
# ═════════════════════════════════════════════════════════════════════════
def _attn(q, k, v, mask, row_mask, scale):
    """`softmax(q·kᵀ·scale + mask) @ v`，q:(B,H,Lq,dk) k/v:(B,H,Lk,dk)。

    `mask`：(B,1,Lq,Lk) 加性掩码（0 / NEG_BIG），4D 广播加法。
    `row_mask`：(B,1,Lq,1) 或 None —— 整行都被掩掉的行输出 0（对齐 PyTorch SDPA
    对全掩码行返回 0 的语义；若用 -inf，ORV 的 softmax 会出 NaN）。
    """
    scores = (q @ k.swap_last2()) * float(scale)              # (B,H,Lq,Lk)
    scores = scores + mask
    p = T.functional.softmax(scores, axis=-1)
    out = p @ v                                               # (B,H,Lq,dk)
    if row_mask is not None:
        out = out * row_mask
    return out


# ═════════════════════════════════════════════════════════════════════════
# 前向
# ═════════════════════════════════════════════════════════════════════════
def csa_forward(x, w, plan, cos, sin):
    """CSA 前向。形状：`x` (B,T,d_model) → (B,T,d_model)。

    `w`：权重 dict（见 `init_weights` / `weights_from_torch_state_dict`）
    `plan`：`make_plan` 的产物；`cos/sin`：`rope_tables` 的全表 (seq_len, rot_dim)。
    """
    cfg = plan.cfg
    B, Tlen = int(x.shape[0]), int(x.shape[1])
    Tc, c = plan.Tc, plan.c
    if (B, Tlen) != (plan.B, plan.Tlen):
        raise ValueError("x 形状 (%d,%d) 与 plan (%d,%d) 不一致。修法：x 或 seg 变了就重建 plan"
                         % (B, Tlen, plan.B, plan.Tlen))
    dev = x.device
    t = plan._tensors(dev)
    d_k, n_q, n_kv, latent = cfg.d_k, cfg.n_q_heads, cfg.kv_heads, cfg.latent
    eps = cfg.norm_eps
    scale = 1.0 / math.sqrt(float(d_k))

    # ── 低秩下投影 + 压缩（池化 → 卷积 → RMSNorm）──
    kv = x @ w["wkv"].transpose(0, 1)                          # (B,T,latent)
    if plan.use_seg:
        pooled = _pool_from_layout(kv, c, plan._n["pool"]["layout"], Tc)
        if plan.single:
            hc = _conv_whole(pooled, w["conv"], cfg.csa_causal_conv)
        else:
            hc = csa_conv_seg_orv(pooled, w["conv"], plan)
    else:
        pad = plan._n["pad"]
        if pad:
            last = kv[:, Tlen - 1:Tlen, :]
            kv_p = T.cat([kv] + [last] * pad, dim=1)
        else:
            kv_p = kv
        pooled = kv_p.reshape(B, Tc, c, latent).mean(axis=2)
        hc = _conv_whole(pooled, w["conv"], cfg.csa_causal_conv)
    hc = _rms(hc, w["ln"], eps)

    # ── 投影出 q / k / v ──
    q = _heads(x @ w["wq"].transpose(0, 1), n_q, d_k)           # (B,n_q,T,d_k)
    k = _heads(hc @ w["wk"].transpose(0, 1), n_kv, d_k)        # (B,n_kv,Tc,d_k)
    v = _heads(hc @ w["wv"].transpose(0, 1), n_kv, d_k)
    if cfg.attn_qk_norm:
        q = _rms(q, w["qn"], eps)
        k = _rms(k, w["kn"], eps)

    # ── RoPE（两个分支各自一套位置；θ/partial 相同，共用同一张表）──
    rv = plan._rope_views(cos, sin)
    q_g = _rope(q, rv["cos_q"], rv["sin_q"], dev)
    k_g = _rope(k, rv["cos_k"], rv["sin_k"], dev)

    if n_kv != n_q:
        rep = n_q // n_kv
        k_g = k_g.repeat_interleave(rep, dim=1)
        v_g = v.repeat_interleave(rep, dim=1)
    else:
        v_g = v

    # ── 全局压缩分支 ──
    out_g = _attn(q_g, k_g, v_g, t["mask_g"], t.get("row_g"), scale)

    # ── 局部精确窗分支（key 是最后 lk 个 chunk，参考实现口径）──
    lk = plan.lk
    k_l = k[:, :, Tc - lk:Tc, :]
    v_l = v[:, :, Tc - lk:Tc, :]
    q_l = _rope(q, rv["cos_q"], rv["sin_q"], dev)
    k_l = _rope(k_l, rv["cos_kl"], rv["sin_kl"], dev)
    if n_kv != n_q:
        rep = n_q // n_kv
        k_l = k_l.repeat_interleave(rep, dim=1)
        v_l = v_l.repeat_interleave(rep, dim=1)
    out_l = _attn(q_l, k_l, v_l, t["mask_l"], t.get("row_l"), scale)

    # ── 合并 + 上投影 ──
    gate = nn_ops.sigmoid(w["gate"])                           # (1,)
    out = out_g + gate * out_l
    out = out.transpose(1, 2).reshape(B, Tlen, n_q * d_k)
    return out @ w["wo"].transpose(0, 1)


# ═════════════════════════════════════════════════════════════════════════
# 权重：随机初始化 / 从 PyTorch state_dict 迁移
# ═════════════════════════════════════════════════════════════════════════
def init_weights(rng, cfg, scale=0.02, device=None):
    """随机初始化一套 CSA 权重（可训练；键名见下）。

    键名与参考实现字段的对应：
        wq  ← q_proj.weight      (n_q*d_k, d)
        wkv ← kv_down.weight     (latent, d)
        wk  ← k_down.weight      (n_kv*d_k, latent)
        wv  ← v_down.weight      (n_kv*d_k, latent)
        wo  ← o_proj.weight      (d, n_q*d_k)
        conv← conv.weight        (latent, k)   ★ 方向已按 ORV 语义翻过
        ln  ← latent_norm.weight (latent,)
        qn/kn ← q_norm/k_norm     (d_k,)       attn_qk_norm=False 时为 None
        gate ← gate (1,)，参考实现从 0 起步（sigmoid(0)=0.5 的局部权重）
    """
    d = cfg.d_model
    def p(*shape):
        arr = rng.normal(0, scale, tuple(int(s) for s in shape)).astype(np.float32)
        return T.tensor(arr, requires_grad=True, device=device)
    w = {
        "wq": p(cfg.n_q_heads * cfg.d_k, d),
        "wkv": p(cfg.latent, d),
        "wk": p(cfg.kv_heads * cfg.d_k, cfg.latent),
        "wv": p(cfg.kv_heads * cfg.d_k, cfg.latent),
        "wo": p(d, cfg.n_q_heads * cfg.d_k),
        "conv": p(cfg.latent, cfg.csa_conv_kernel),
        "ln": T.tensor(np.ones(cfg.latent, np.float32), requires_grad=True, device=device),
        "gate": T.tensor(np.zeros(1, np.float32), requires_grad=True, device=device),
        "qn": None, "kn": None,
    }
    if cfg.attn_qk_norm:
        w["qn"] = T.tensor(np.ones(cfg.d_k, np.float32), requires_grad=True, device=device)
        w["kn"] = T.tensor(np.ones(cfg.d_k, np.float32), requires_grad=True, device=device)
    return w


def weights_from_torch_state_dict(sd, cfg, device=None):
    """把 PyTorch 参考模块的 state_dict（numpy/张量皆可）映射成 ORV 权重 dict。

    ★ `conv.weight` 在 PyTorch 里是 (latent, 1, k)，且方向与 ORV 的 (latent, k) **相反**
    （见 `csa_conv_causal_orv` 的说明）⇒ 这里 reshape 后翻转最后一维。
    """
    def get(name):
        v = sd[name]
        return np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v,
                          np.float32)

    def p(arr):
        return T.tensor(np.ascontiguousarray(arr, np.float32),
                        requires_grad=True, device=device)

    conv = get("conv.weight")
    k = conv.shape[-1]
    w = {
        "wq": p(get("q_proj.weight")),
        "wkv": p(get("kv_down.weight")),
        "wk": p(get("k_down.weight")),
        "wv": p(get("v_down.weight")),
        "wo": p(get("o_proj.weight")),
        "conv": p(np.flip(conv.reshape(cfg.latent, k), axis=1).copy()),
        "ln": p(get("latent_norm.weight")),
        "gate": p(np.asarray(get("gate"), np.float32).reshape(1)),
        "qn": None, "kn": None,
    }
    if cfg.attn_qk_norm:
        w["qn"] = p(get("q_norm.weight"))
        w["kn"] = p(get("k_norm.weight"))
    return w
