# 移植计划：把 PyTorch 的 mem-efficient 注意力算法搬到 ORV

> 依据：读 `D:\DSH-Workspace\pytorch-src\aten\src\ATen\native\transformers\cuda\mem_eff_attention\`
> 的 `kernel_forward.h`（56 KB）与 `kernel_backward.h`（103 KB）。
> 这是 PyTorch 在 **V100（SM70）** 上实际使用的注意力后端（V100 没有 flash-attn，只有这个）。

## 1. PyTorch 到底怎么做的（源码事实）

前向 `kernel_forward.h`：

```
一个 CTA 负责一个 query 块：  query_start = blockIdx.x * kQueriesPerBlock
对 key 块循环（kKeysPerBlock）：
    S      = Q_blk · K_blkᵀ · scale          # 形状 (C_q, C_k)，**永不物化 (S,S)**
    m_new  = max(m, rowmax(S))               # 行最大值在线更新
    l      = l·exp(m − m_new) + Σ exp(S − m_new)
    O      = O·exp(m − m_new) + softmax(S − m_new)·V_blk
产出：O（每行 d 维）+ LSE（每行 1 个标量，`kAlignLSE = 32` 对齐，供反向分块用）
```

反向 `kernel_backward.h`：**用 O 与 LSE 重算 S**，不保存 P；对每个 (query 块, key 块) 重算后
累积 dQ/dK/dV。所以显存与 `S²` 无关，只有 `O(S·d)` 与 `O(S)`。

三个关键设计点：

| 点 | 做法 | 为什么重要 |
| --- | --- | --- |
| 块大小 | `kQueriesPerBlock` / `kKeysPerBlock` 模板参数，按 arch 选 | 决定 SM 占用与 smem 用量 |
| LSE 对齐 | `kAlignLSE = 32` | 反向按 32 行一组读 LSE |
| 不存 P | 反向重算 | 显存从 O(S²) 降到 O(S·d) |

### 1.1 实测的块大小（读 `kernels/generate_kernels.py`，不是猜的）

反向核生成器里：

```python
bi_values = [64]
if sm >= 80 or (sm >= 70 and is_half):
    if max_k > 64: bi_values.append(128)          # V100 用 fp16 且 head_dim>64 时可用 128
bj = 128 if (preload_mmas and max_k > 64) else 64 # preload_mmas 要求 sm >= 80
```

⇒ **V100（SM70）上：queries/block = 64 或 128，keys/block = 64**。
（`block_j = 128` 只在 `preload_mmas` 时出现，而它要求 `sm >= 80`。）

另外 `sdp_utils.cpp:493` 的 `check_flash_attention_hardware_support` 明确写着
flash 只支持 `[sm80, sm121]`，并存在 `less_than_sm80_mem_efficient_dtypes` 白名单
（第 1252 行）——**V100 走的就是 mem-efficient，且精度组合受白名单限制**。

**所以移植时块大小应取 64/64（或 fp16 下 128/64），不要用我先前拍的 256。**

## 2. 在 ORV 里怎么做（不需要写 CUDA）

ORV 有张量运算与 autograd，但**没有块级融合**。所以移植的是**算法**，用现有算子表达：

### 前向（`no_grad` 下执行，只留 O 与 LSE）

```python
for qb in query_blocks(S, Cq):
    m = full(-inf); l = zeros(Cq); O = zeros(Cq, d)
    for kb in key_blocks(S, Ck):                     # 因果时只到 qb 末尾
        S_b = (Q[qb] @ K[kb].T) * scale              # (Cq, Ck)
        S_b = mask(S_b, qb, kb)                      # 因果/块对角
        m_new = max(m, S_b.max(-1))
        e = exp(S_b - m_new[:, None])
        l = l * exp(m - m_new) + e.sum(-1)
        O = O * exp(m - m_new)[:, None] + e @ V[kb]
        m = m_new
    save(O[qb], m + log(l))                          # LSE
```

**显存**：每步只有 `(Cq, Ck)` 与 `(Cq, d)`；全程 `O(S·d)`。

### 反向（重算，不存 P）

```python
for qb, kb:
    S_b = recompute(S_b, Q, K, LSE, qb, kb)          # 用 LSE 归一化
    P = exp(S_b - LSE[qb][:, None])
    dV[kb] += P.T @ dO[qb]
    dP = dO[qb] @ V[kb].T
    dS = P * (dP - (dP * P).sum(-1, keepdim))        # softmax 的反向
    dQ[qb] += (dS @ K[kb]) * scale
    dK[kb] += (dS.T @ Q[qb]) * scale
```

`dQ/dK/dV` 用已有的 `scatter_add`/索引累加即可。

## 3. 为什么这能同时解决显存与速度

| | 现在（物化）| 移植后（分块在线）|
| --- | --- | --- |
| 峰值 | `B·H·S²`（S=2048、8 头 ⇒ 134 MB/层·次）| `B·H·Cq·Ck`（256×256×4B = 0.26 MB/块）|
| 反向 | 存 P 并回读 | 重算，无 P |
| 速度 | 一次性大矩阵，但**被显存带宽拖住** | 小块 matmul 命中 L2/smem，PyT 实测更快 |

## 4. 实施顺序（每步可验证）

1. **前向数值对齐**：与 `attention_orv` 在 S=512/2048 上比输出，相对误差 < 1e-5
2. **只前向的显存**：确认 peak_live ≈ O(S·d) + 权重，与 S 近似线性而非平方
3. **反向数值对齐**：与现有 autograd 的梯度比（同输入），相对误差 < 1e-4
4. **完整训练步**：接进 `v3_orv_train.py`，用同一套精确计数重跑 seq 512/1024/2048 三档
5. 与 PyTorch（已开检查点）同表对比，看显存倍数与速度比

## 5. 风险与对策

| 风险 | 对策 |
| --- | --- |
| Python 循环的启动开销（块数 × 层数 × 步数）| 先测 Cq=Ck=256 的块数（S=2048 时 64 块/层），若开销大再试 512 |
| 因果掩码的块分类（全可见/对角/全屏蔽）| 对角块用掩码，其余两块直接跳过或直算，避免无谓计算 |
| LSE 数值稳定性 | 与源码一致：`m` 在线更新 + `l` 同步缩放 |

## 6. 实现进度（诚实版）

| 步骤 | 状态 | 证据 |
| --- | --- | --- |
| 前向在线 softmax | ✅ 数值通过 | 与整体注意力 **1.04e-07**（cq=ck=128 与 256 都过）|
| 前向显存随 S 线性 | ✅ **已实测** | 见下表 |
| 反向重算 | ⬜ 未做 | — |
| 训练步三档重测 | ⬜ 未做 | — |

### 6.1 显存曲线（单层注意力，块 64×64，每档单独进程）

| S | flash `peak_live` | 物化 `peak_live` | 比值 | flash 耗时 | 物化耗时 |
| --- | --- | --- | --- | --- | --- |
| 512 | **27 MiB** | 48 MiB | 0.56× | 0.04 s | 0.01 s |
| 1024 | **48 MiB** | 139 MiB | 0.35× | 0.11 s | 0.01 s |
| 2048 | **89 MiB** | 472 MiB | **0.19×** | 0.38 s | 0.02 s |

**这就是要的形状**：flash 随 S 近似线性（27→48→89），物化超线性（48→139→472）。
S=2048 时省 5.3× 显存 —— 与 PyTorch mem-efficient 的设计目标一致。

**但速度是 Python 循环的代价**：0.38 s vs 0.02 s（慢 19×）。其中很大一部分来自每块
`sb.to_numpy()` 的主机往返与 numpy 侧掩码；这两处改成设备侧后能显著缩小差距。

**仍未解决**：反向重算。没有它，这个前向不能用于训练（无法反传）。

### 6.2 设备侧版本：NaN 的根因（已修）与速度现实

两次设备侧尝试都吐 NaN。四个探针逐个排除了"原语不可靠"这个方向：

| 怀疑对象 | 结果 |
| --- | --- |
| `T.cat(dim=2)` + `max(axis=-1)` 组合 | ✅ 与 numpy 逐值一致 |
| 四维切片 `q[:, :, a:b, :]` | ✅ 一致 |
| **切片视图上 reshape**（形状对但值错是最隐蔽的坑）| ✅ 值也一致 |
| `exp(-1e30)` / `log(0)` | ✅ 得 0 / `-inf` |

真凶是**数值边界的算术**：

```
掩码用 -inf，初始运行最大值 m = -1e30（有限）
整行被掩 ⇒ m_new = max(-1e30, -inf) = -1e30
sb − m_new = −inf − (−1e30) = +inf  ⇒  exp(+inf) = inf  ⇒  l = inf, acc = inf  ⇒  inf/inf = NaN
```

主机版靠 `np.where(np.isfinite(sb), e, 0)` 这层保护才没炸；设备版丢了它。
**修法：掩码用有限哨兵 `NEG = -1e9`**，被掩位置 `sb − m_new ≈ -1e9 ⇒ exp ≈ 0`，既有 0 又不产生 inf。

修好后设备版数值 **1.19e-07**，与主机版、物化路径都一致。

### 6.3 速度：Python 层追不平融合内核（实测）

| 版本 | 数值 | 耗时（S=1024，单层注意力）|
| --- | --- | --- |
| 主机侧块化 | 1.49e-07 | 0.112 s |
| 设备侧块化（已修）| 1.19e-07 | 0.184 s |
| 物化（融合 op）| — | **0.003 s** |

**结论：算法性省显存已达成（线性 vs 超线性，S=2048 省 5.3×），但速度差 60 倍。**
根因不是算术而是**启动次数**：Python 每层上千次小 kernel 启动，不可能追平一次融合启动。
⇒ **速度持平必须走 CUDA 内核**，不要再在 Python 循环里优化。
