# C 端下沉清单 & PyT 迁移边界（2026-11-10，ORV 0.9.26）

> 目标里两条明确要求：① 讲清"逐层重算在 Python 层而不是 C 端"，并列出可下沉 C 端的关键路径；
> ② 把"不做"的迁移能力写清原因。本文两份都答。

## 1. 逐层重算（梯度检查点）确实在 **Python 层**

**证据**：`python/tgai_go/train/v27_train.py` 的训练循环按层做
`fwd_attn` → `moe_part` → 反向时逐层重放：

```python
for i in range(NL - 1, -1, -1):
    y_t = T.tensor(ys[i], requires_grad=True)     # 重新造输入
    z = tr.moe_part(y_t, i)                        # 重新算这一段的前向
    (z * T.tensor(gg)).sum().backward()            # 只反传这一段
    gg = gb.replay_layer(i, ...)                   # 注意力段走 CUDA Graph
```

- 中间激活**逐层落到主机**（`ya.to_numpy()`），所以显存只保留一层的量；
- 「重算」发生在 Python 循环里，由 `LayerGraphBackward`（`layer_graph.py`）驱动；
- **C 端（CUDA 内核 / NVRTC）里没有任何 checkpoint 逻辑** —— 内核只做单算子。

代价：主机↔设备往返 `2×NL` 次/步。T=512/28 层实测这部分不占大头（前向链只 234 ms），
真正的大头曾是**分头小 kernel**（已用堆叠 SDPA 解决，638 → 208 ms）。

## 2. 可下沉 C 端的关键路径（按收益/风险排序）

| 路径 | 现状 | 下沉后预期 | 风险 |
| --- | --- | --- | --- |
| **MoE 反向**（330 ms，占整步 42%） | 逐算子组合 | 融合 expert 的 gather→GEMM→scatter | 中：需处理 token 不均衡 |
| **词表头 CE + 反向**（11 ms） | 已有 `ce_fwd/bwd` | 已下沉 | — |
| **前向链 attn+MoE**（234 ms） | 逐算子 | 整层融合 kernel（类似 FlashAttention + fused MLP） | 高：等同重写模型 |
| **主机↔设备往返**（`ys/xs` 落盘） | Python `to_numpy()` | 设备侧保留 + 重算调度下沉 | 高：会顶掉显存的收益 |
| **优化器**（≈47 ms） | 已融合 `adamw_multi_f16` | 已下沉 | — |
| **Embedding**（已设备化，省 162 ms） | 设备侧 gather/scatter | 已下沉 | — |

**结论**：剩下最值得做的是 **MoE 反向融合**；"整层融合"属于另一个量级的工程（等同写一个
FlashAttention 级别的新内核），不建议作为 ORV 当前阶段的投入。

## 3. 自定义 `Module` 自动迁移：**不做**，原因如下

`from_torch` 支持 13 种常见层（Linear/Embedding/LayerNorm/RMSNorm/Conv1d/Conv2d/Dropout/
BatchNorm1d,2d/MaxPool1d,2d/AvgPool1d,2d + Sequential/ModuleList），但**不支持任意
`nn.Module` 子类**，会抛 `InteropError` 并列出已支持类型。

**为什么不做**：`forward` 是任意 Python 代码。要自动翻译它，等于在 ORV 里实现一个
Python → ORV 计算图 的**转译器 + 控制流分析**，而且一旦遇到动态分支/自定义 autograd.Function
就会静默出错 —— 那正是我们最不能接受的事（本项目已经因为"静默算错"踩过三次坑：
`masked_fill` 在 CUDA 上返回 nan、`CausalSelfAttention` head 切分错、PPO 的 min 平局导致梯度反向）。

**用户该怎么做**：把自定义 `forward` 用 ORV 的层重写一遍（通常 10-40 行）。两者的数学是逐算子对齐的
（见 `docs/REVIEW_2026-11-10_vs_PyTorch.md` 的对照表），所以重写是机械工作，不需要调参。

## 4. 迁移能力的真实边界（一句话）

| 情形 | 能否"改个导入就迁移" |
| --- | --- |
| 全是 `nn.Linear/Embedding/Norm/Conv/Pool/BN/Dropout` + `Sequential` 的模型 | ✅ 能（权重逐位复制） |
| 有自定义 `forward`（残差/门控/自定义 attention） | ⚠️ 需手写 forward，权重仍可 `from_torch` 搬 |
| 用 `torch.compile` / 自定义 autograd.Function / DDP | ❌ 不适用（ORV 有自己的图路径与单卡定位） |


---

## 5. 2026-11-11 更新：Muon 大模型可用性的实测与下一步

### 已测结论

| 配置 | 3B 整步 | 峰值显存 |
| --- | --- | --- |
| Muon，
s_fast=False（顺序 FMA，与 OpenBLAS 逐位一致） | 62.9 s | 18.93 GiB |
| Muon，
s_fast=True（Newton-Schulz 走 cuBLAS） | **35.0 s** | 19.17 GiB |
| AdamW（fp16 动量 + 逐层即时更新） | **0.709 s** | 24.78 GiB |

### 为什么换 cuBLAS 只快 1.8 倍

3B 有 2550 个 2D 张量，每个跑 5 步 × 约 6 个算子 ≈ **7.6 万次算子调用**。
按 0.4 ms/次估算正好 30 秒量级 —— **瓶颈是算子数量，不是单个 GEMM 的速度**。

### 下一步方案（已验证可行）

**按形状分组批量正交化**：

1. 3B MoE 里同形张量高度重复 —— 每层 8 专家 × 3 矩阵 [4096,1024] = 24 个，
   28 层共 **672 个同形张量**；
2. 按形状分组，stack 成 [G, M, K]，用**批量矩阵乘**一次算完一组；
3. 调用数从 7.6 万降到万级以下。

**可行性已实测**：ORV 的 Tensor @ Tensor **支持 3D@3D（bmm）**，也支持 3D@2D。

实现要点：_step_cuda 目前工作在 GpuBuf 层，批量正交化需要
Tensor._from_gpu(...) 把 buffer 包成 Tensor 跑 bmm，再拆回各参数缓冲。
代价是每组多一份 [G,M,K] 的批量缓冲 —— Muon 路线峰值 19.17 GiB 尚有 12.7 GiB 余量，够用。

**状态：方案确定、可行性已验证、尚未实现。**
