# 通用能力缺口（不是模型适配）

> 原则：**PyTorch 能表达的，ORV 就应该能表达。** 所以对标的单位是**能力**，不是某个模型。
> 21e 只是**测试用例** —— 用它暴露缺口，但不为它特判。
>
> 方法：**像普通用户那样只用公开 API 搭**，凡是要绕路的地方就是缺口。
> 每个缺口必须用**通用形式**修，修完之后任何同结构的用户模型都自动受益。

## 1. 审计结果（脚本：`tools/audit_generality.py`）

| 能力 | ORV | 说明 |
| --- | --- | --- |
| **每层可指定不同 `d_head`** | **❌ 缺** | `nn.CausalSelfAttention(d_model, n_heads)` 只能推出 `d_head = d_model/n_heads` |
| **SDPA 接受任意 mask（bool/float）** | **❌ 缺** | 只有 `causal` / `is_causal`；没有 `attn_mask` |
| **AMP 把激活存成 fp16/bf16** | **❌ 缺** | 只 cast GEMM 输入，激活全程 fp32 |
| `RMSNorm` 的 γ 可学习 | ✅ 有 | |
| 大专家数 MoE（36 专家 top-6）| ✅ 有 | `nn.MoE(d_model, d_ff, 36, top_k=6)` |
| 分块 CE（避免 `[B,T,V]`）| ✅ 有 | `functional.linear_cross_entropy` |
| LayerScale | ⚠️ 无专层，但**不算缺口** | 一个可学习标量乘子，`nn.Parameter` 手工即可 |

## 2. 三个缺口的通用描述

### 缺口 A：头数与头维被绑死

PyTorch 里两者**独立**：

```python
qkv = nn.Linear(d_model, n_heads * d_head)   # 任意组合
```

ORV 里 `d_head = d_model // n_heads` **强制**。当 `n_heads * d_head != d_model` 时
（例如 d_model=1280、10 头 × 96 = 960），**ORV 根本表达不了**，用户只能手写整套注意力。

**通用修法**：`nn.CausalSelfAttention(d_model, n_heads, d_head=None, ...)`，
`d_head` 给了就用它（投影到 `n_heads*d_head`，输出再投回 `d_model`）。

**连带影响**：`sdpa_lowmem` 目前要求 `dh==64`，正因为 `dh` 被绑死才只能挑一个值硬编。
`d_head` 一旦可指定，lowmem 必须支持**任意 dh**（至少覆盖常见 64/96/128），
否则用户一换头维就静默退回全量注意力。

### 缺口 B：注意力只支持 causal

PyTorch 的 `scaled_dot_product_attention(q,k,v, attn_mask=...)` 接受
`[S,S]` 或 `[B,H,S,S]` 的 bool/float 掩码。真实场景里这非常常用：
文档打包（同文档才互相看）、prefix-LM、自定义 padding、稀疏掩码。

**ORV 只有 `is_causal`** ⇒ 用户必须自己物化整张注意力再掩码，**既慢又费显存**。

**通用修法**：`functional.scaled_dot_product_attention(..., attn_mask=None)`，
接受 bool/float；bool 掩码走融合核（这是打包训练的关键路径）。

### 缺口 C：AMP 不省激活显存

`torch.autocast` 把**激活**存成 fp16（只在需要时 cast 回 fp32）；
ORV 的 `T.amp()` 只把 **GEMM 输入** cast，参数/激活/梯度全程 fp32。

⇒ 同样形状下 ORV 的激活显存约 **2×**。这是「P100 16GB 训不了」的第一原因，
也是「ORV 显存比 PyTorch 多 1.2–1.9×」的主要来源（见 `BENCHMARK_H2H_2026-11-12.md`）。

**通用修法**：真正的 AMP —— 激活存储用 fp16/bf16，算子边界按需 cast，
SDPA/FFN 的中间量优先用低精度。这条在 `MULTIVENDOR_ROADMAP.md` §6 里已列为待做项。

## 3. 三者之间的关系

```
缺口 A（d_head 可指定）
   └─ 迫使 lowmem 支持任意 dh  ──→ 影响「大 dh 无法省显存」
缺口 B（任意 mask）      ──→ 文档打包/prefix-LM 用户无法高效训练
缺口 C（真 AMP）        ──→ 激活显存 2×  ──→ 16GB 卡放不下
```

**C 是显存问题的主因，A/B 是"能不能表达"的问题。** 三者都必须在**通用层**修，
修完之后 21e 自动能跑 —— 而不是为 21e 加分支。

## 4. 验收方式（这才是"通用"的证明）

**不许改 ORV 去适配某个模型。** 正确做法：

1. 写一个**用户态**的模型实现（只用公开 API，`examples/` 下，不碰核心）。
2. 如果搭不出来 ⇒ 记录**卡在哪一条 API 上**，那就是缺口。
3. 缺口按通用形式修（如上三节）。
4. 修完后同一个用户态实现**不加任何改动**就能跑 ⇒ 通用性成立。

**判断标准**：用户态实现里**不允许出现 `if 模型名 == ...`**，
也不允许调用任何非公开接口。出现即说明缺口没修好。

## 5. 优先级

| # | 缺口 | 为什么先做 |
| --- | --- | --- |
| 1 | **C 真 AMP** | 唯一能追平 torch 显存的杠杆；决定 16GB 能不能训 |
| 2 | **A d_head 可指定 + lowmem 任意 dh** | 决定"非标准头维"的模型能不能表达 |
| 3 | **B 任意 mask** | 决定文档打包类训练能不能高效 |

**注意**：这三项都**不是** 21e 专属 —— 任何用户模型都可能用到。
21e 只是恰好同时踩中三条，所以它是一个好的验收用例。
