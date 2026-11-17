# ORV 长上下文方案（设计，2026-11-17）

> 目标见 `goal-8d7c1b65`：把 CSA 注意力的显存从 O(seq²) 降到 O(seq)，让 ORV 具备 32K/64K 训练能力。

## 一、实测到的结构（读源码确认，非推测）

`v3_csa.py: csa_forward` 有两个注意力分支：

| 分支 | 调用 | q 形状 | k/v 形状 | scores 形状 | 复杂度 |
|---|---|---|---|---|---|
| **全局压缩** | `L835 _attn(q_g,k_g,v_g, mask_g, row_g, scale)` | (B,n_q,**T**,d_k) | (B,n_q,**Tc**,d_k) | (B,n_q,**T**,**Tc**) | **O(T²/c)** |
| 局部窗 | `L847 _attn(q_l,k_l,v_l, mask_l, row_l, scale)` | (B,n_q,T,d_k) | (B,n_q,**lk**,d_k) | (B,n_q,T,lk) | O(T)（lk 常数）|

`Tc = T/c`（`c=4`）。**二次项只来自全局分支。**

`_attn`（L758）本身物化三份 O(T·Tc)：

```python
scores = (q @ k.swap_last2()) * scale    # ①
scores = scores + mask                   # ②（又一份）
p = T.functional.softmax(scores, axis=-1)# ③
out = p @ v
```

**掩码也是物化的**：

* `_mask_global_fixed`（L415）：`np.ascontiguousarray(np.broadcast_to(m,(B,1,T,Tc)))`
  —— 结构上只是"chunk 粒度的因果"（`ik <= iq // c`），却实打实复制了一份 `B·T·Tc·4` 字节。
  T=16384 时 = **268 MB**；T=65536 时 = **4.3 GB**。
* `_mask_global_seg`（L440）：块对角（按文档边界）+ 因果，同样是 O(T·Tc)。

⇒ **要真正做到 O(seq)，两件事都要做：分块算 attention + 不再整块物化掩码。**

## 二、方案：按 **query 行**分块（数学逐行独立，结果等价）

softmax 是**逐行**的 ⇒ 把 q 切成 `cq` 行一块，逐块 `_attn`，拼接结果，
与整块计算**数学等价**（同一行参与的计算完全相同）。

```python
def _attn_qchunk(q, k, v, mask, row_mask, scale, cq=1024):
    Lq = q.shape[2]
    outs = []
    for i0 in range(0, Lq, cq):
        i1 = min(i0 + cq, Lq)
        outs.append(_attn(q[:, :, i0:i1], k, v,
                          mask[:, :, i0:i1] if mask is not None else None,
                          row_mask[:, :, i0:i1] if row_mask is not None else None,
                          scale))
    return T.cat(outs, dim=2)
```

* scores/p 的峰值从 `(B,n_q,T,Tc)` 降到 `(B,n_q,cq,Tc)` ⇒ **对 T 线性**
* 反向：cat 的反向是切片，**没有额外常驻**；每个 chunk 的 p 在反向时按需重算或由 autograd 保存
  （注意：这一版仍会保存每个 chunk 的 p，但总量还是 O(T·Tc)！）

⚠️ **关键点**：前向峰值降下来了，但**反向**若保存全部 `p`，常驻仍是 O(T·Tc)。
所以分块必须做在**算子内部**（fused），而不是用 Python 循环把 autograd 图拆成 n 个 chunk。
两种做法：
  a. **推荐**：写一个分块注意力内核（前向 online-softmax，反向重算 online-softmax）⇒ 真 O(T)；
  b. 过渡：Python 分块 + 立即 detach 中间量（失去精确梯度，不可接受）。

⇒ **所以本目标的核心交付其实是一个 CSA 专用的分块注意力算子**，
而不是"Python 里循环调用 `_attn`"。

## 三、分两步走（降低风险）

**第一步（有独立价值，先做）**：把**掩码**从"整块物化"改成"按 chunk 惰性构造"。
`_mask_global_fixed` 的结构极简（`ik <= iq//c`），完全可以不物化：
给 `_attn` 传 `causal_chunk=c` 的语义，在内核里按需算可见性。
这一步单独就能砍掉 `B·T·Tc·4` 字节（16K 时 268 MB、64K 时 4.3 GB）。

**第二步（核心）**：写 CSA 专用的分块注意力内核 `csa_attn_fwd/bwd`：
* 前向：外层循环 chunk（key 方向），online softmax（不存 p）
* 反向：重算 attention 概率（不存 p），只存 q/k/v 与输出
* seg 掩码：用"文档区间表"在内核里判断可见性（不建矩阵）

## 四、验收（与目标一致）

| 项 | 判据 |
|---|---|
| 显存阶数 | 超小模型 8192/16384/32768 上 p ≤ **1.15** |
| 64K 可跑 | d=128/2 层在 65536 完成 3 步，有 loss |
| 数学 | `faithful_parity` ≤1e-5；814 passed |
| 端到端 | 4 层/seq512 step0 loss 与基线 11.713757 相对差 ≤1e-6 |
| 可回退 | `TG_CSA_ATTN=materialized` 走旧路径，两条路径都有测试 |

## 五、参考：PyTorch 的对应做法（"抄作业"的具体内容）

| PyTorch 机制 | 我们要做的对应物 |
|---|---|
| `F.scaled_dot_product_attention` 自动选 flash/mem-efficient 后端 | CSA 分块内核 + 开关 |
| flash 前向 online softmax，不存 `p` | 同上 |
| flash 反向重算 `p`（`save_for_backward` 只存 q/k/v/o + logsumexp）| 同上 |
| 自定义 additive mask 会**退化**到 math 后端（性能差但能跑） | 我们的 seg 掩码可用区间表保持高效 |

**注**：PyTorch 在"带任意 additive mask"时也会退回物化路径 —— 所以我们做的
"区间表 + 分块"其实**比 PyTorch 的通用路径更好**，这也是我建议不直接调
`v3_flash_op.flash_attn`（它只处理 causal）的原因。
