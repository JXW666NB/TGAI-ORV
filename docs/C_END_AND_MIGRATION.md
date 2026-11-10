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
