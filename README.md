# TGAI ORV — Origin Requiem Vehicle（原生之初）

<p align="center">
  <strong>用 PyTorch 的写法，在不装 PyTorch 的机器上训练你自己的模型。</strong><br>
  <em>Write like PyTorch. Train without it.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.9.26-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.9+-green" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="license">
  <img src="https://img.shields.io/badge/tests-553%20passed-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/CUDA%20kernels-101-blueviolet" alt="kernels">
</p>

---

## 一句话：我为什么用 ORV 而不用 PyTorch？

**因为你的显卡不够大，而 ORV 能在同样的卡上把模型练下去 —— 而且算出来的数是一样的。**

三件事同时成立，且都有实测：

| 你的担心 | 实测答案 |
| --- | --- |
| "数学对齐了吗？别练完发现是错模型" | 与 PyTorch 逐步 loss **相对差 1.44e-07**；真实 Qwen2.5-0.5B 权重与 HuggingFace **相对差 6.29e-06** |
| "能微调热门模型吗？" | Qwen2.5-0.5B（GQA 14:2、qkv bias、RoPE θ=1e6）**跑通了，与 HF 逐位对上** |
| "能训练我自己的模型吗？" | 任意 `nn.Module` 都能跑；10 行代码从语料开始（见[快速入门](#快速入门从零到能训练只要四步)） |
| "速度如何？" | 4/8/14 层 seq512 比 PyT **快 1.41~1.74×**；大 GEMM 与 PyT AMP **持平略快** |
| "省显存是不是偷了精度？" | **没有**。省的是"优化器状态"和"激活"，不是计算精度 —— 见[为什么省显存](#为什么省显存而不是省精度) |

**ORV 不是 PyTorch 的封装。** 101 个 CUDA 内核在 NVRTC 里现编现用，autograd 自己推，AdamW 自己更。
`pip install tgai-orv` 之后，你的机器上没有 `torch`，也照样训练。

---

## 快速验证：改一行就换掉 PyTorch

```python
# import torch as T                 # 旧写法
import tgai_go.train as T           # 新写法 —— 其余代码基本不用改
```

`Tensor` / autograd / `nn.Module` / `nn.Linear` / `optim.AdamW` / `functional` / `DataLoader` /
`state_dict` —— **名字一样，语义一样，只是不再是 PyTorch。**

### 与 PyTorch 的差异（必须知道的 8 条）

不是"完全兼容"，下面这些是真的不一样，写代码前请看：

| # | 差异 | 影响 / 怎么办 |
| --- | --- | --- |
| 1 | **单卡**，没有 DDP/FSDP | 多卡训练不在范围内；`dist.py` 只有雏形 |
| 2 | 自定义 `nn.Module` 子类的 `forward` **不能自动迁移** | 权重可以 `from_torch` 搬，`forward` 要用 ORV 的层重写一遍（通常 10~40 行） |
| 3 | `Tensor.max(dim=)` 只返回值，**不返回索引元组** | 需要索引就用 `argmax` 配合 |
| 4 | CUDA 上 `permute/reshape` 是**真拷贝**（PyT 是视图） | 大张量频繁 permute 会多占显存；模型代码里尽量少用 |
| 5 | `Tensor.to(device)` / `.cuda()` / `.cpu()` 是**叶子级搬运，会断开 autograd 链** | 只在数据搬运用；不要在 forward 中间调用 |
| 6 | 没有 `torch.compile` 的对应物 | 想加速用 `T.amp()` + 逐层 CUDA Graph（见下） |
| 7 | AMP 的语义不同：ORV 参数/激活/梯度**全程 fp32**，只 cast GEMM 输入 | 所以**不需要 GradScaler**（留着它是为了 PyT 代码能不改） |
| 8 | 没有 `torchvision` / `torchaudio` / 生态包 | 只做训练核心，不做模型动物园 |

---

## 数学对齐（这是最要紧的一条）

**同一个模型、同一份权重、同一个种子，ORV 与 PyTorch 各训 8 步，逐步比 loss：**

```
最大相对差 = 1.44e-07        （判据 1e-4，实测好 700 倍）
```

逐算子对照（ORV vs torch 同类实现，全部实测）：

| 算子 / 路径 | 相对差 |
| --- | --- |
| gather / scatter、CUDA Graph 重放的梯度、融合 fp16-AdamW | **0.00e+00**（逐位相同） |
| `masked_fill`、`repeat_interleave` | 0.00e+00 |
| LayerNorm | 4.38e-08 |
| `sqrt` / `log` / `gelu` | 5.92e-08 / 7.50e-08 / 7.78e-08 |
| `tanh` / `std` | 2.39e-07 / 2.02e-08 |
| Conv2d / 分组卷积 | 2.39e-07 |
| `CausalSelfAttention`（前向+反向） | 5.02e-08 |
| LR 调度（cosine/linear/warmup） | 4.55e-15 |
| **fp16 GEMM（AMP）** | 2.78e-04 ~ 2.88e-04（**与 torch AMP 同量级**，这是 fp16 本身的精度，不是我们的误差） |

**真实外部权重**：Qwen2.5-0.5B-instruct（结构与我们不同）在你自己的张量算子上跑，与 HuggingFace
输出相对差 **6.29e-06**（判据 1e-5）。

---

## 能不能微调热门模型？

**能，已验证到 0.5B 规模。** 拿真实 HF 权重的路径：

```python
import tgai_go.train as T

# 1) 把 torch / HF 的权重搬进来（逐位复制，不改数值）
model = T.from_torch(hf_model)          # 支持 Linear/Embedding/LayerNorm/RMSNorm/
                                        # Conv1d/2d/Dropout/BatchNorm1d,2d/池化/Sequential/ModuleList
# 2) 用 ORV 的训练循环
hist = T.fit(model, data, steps=2000, lr=2e-5)
```

已验证：**Qwen2.5-0.5B-instruct**（GQA 14:2 + qkv bias + RoPE θ=1e6）与 HF 逐位对齐；
LLaMA 结构（RMSNorm+RoPE+SwiGLU+GQA）20 步训练与 PyTorch 孪生 loss 相对偏差 **1.59e-05**。

> ⚠️ **诚实边界**：7B 以上的热门模型我们没有显存实测过。0.5B 是验证过的上限，
> 更大的规模需要你自己按下面的显存表估算。

---

## 速度（与 PyTorch 同结构、同条件、同卡）

**V100 32GB，seq=512，逐层反向：**

| 层数 | ORV | PyTorch | 结论 |
| --- | --- | --- | --- |
| 4 层 | **78.1 ms** | 135.6 ms | ORV 快 **1.74×** |
| 8 层 | **187.5 ms** | 264.1 ms | ORV 快 **1.41×** |
| 14 层 | **327.6 ms** | 484.9 ms | ORV 快 **1.48×** |
| 28 层（真 V2.7） | ~2038 ms | 2056 ms | **基本持平**，见下方说明 |

**大 GEMM（2048×2048×6144）**：

| 口径 | 耗时 |
| --- | --- |
| ORV fp32 | 4.33 ms |
| **ORV AMP（fp16）** | **0.67 ms（6.45×）** |
| PyTorch AMP | 0.71 ms |

**ORV 自己的加速（同一份代码，开优化前后）**：

| 场景 | 优化前 | 优化后 | 倍数 |
| --- | --- | --- | --- |
| T=128 整步 | 1843 ms | **721 ms** | 2.56× |
| T=512 整步（28 层） | 2038 ms | **754 ms** | 2.70× |
| 单层注意力前向（seq=256） | 9.8 ms | **2.7 ms** | 3.66× |

提速来自三处，都可在代码里看到：**逐层 CUDA Graph**（注意力反向 638→208 ms）、
**注意力堆叠实现**（消灭 16 次小 kernel）、**融合 fp16-AdamW**（8×）。

> 28 层"持平"是实话：层数越多，ORV 的逐层重算（Python 层的梯度检查点）开销占比越大。
> 754 ms 是开了图之后的结果；不开图是 2038 ms 与 PyT 打平。

---

## 显存：为什么省显存，而不是省精度

这是最容易被误解的一条，所以说清楚。

### 省在哪（两项，都与计算精度无关）

| 手段 | 省下什么 | 代价 |
| --- | --- | --- |
| **逐层重算**（梯度检查点） | 不保存全部 28 层的激活，只留一层 | 反向时多算一遍前向（时间换显存） |
| **fp16 优化器动量**（可选） | 2.97B 参数的 m/v 从 22.2 GiB 降到 **11.1 GiB** | Adam 的更新是 `m/√v`（尺度不变），fp16 误差在分子分母上大部分抵消 |

**计算本身仍然是 fp32**：参数、激活、梯度全程 fp32，AMP 只是把 GEMM 的输入 cast 成 fp16
再用 fp32 累加。所以**不存在"用低精度换显存"**这件事。

### fp16 动量真的不掉质量吗（实测）

| 指标 | 结果 |
| --- | --- |
| 参数偏差 | 4.6e-05 |
| 更新方向余弦相似度 | **1.000000** |
| loss 偏差 | 3.2e-05 |

### 实测显存（V100 32GB，真实 V2.7 28 层 2.97B）

| 项目 | 数值 |
| --- | --- |
| 装载 28 层后的权重 | **11.39 GiB**（空闲 20.47 / 31.86 GiB） |
| 优化器状态（fp16 动量，实测增量） | **+11.1 GiB** |
| 优化器状态（fp32 动量，= PyTorch AdamW） | 约 **22.2 GiB**（算术确定，见下） |

### ⚠️ 一个我**还没有验证**的结论（不要当成卖点传播）

"V100 单卡 32GB 能全参 AdamW 训练 3B" —— **这一条我目前只能给出算术，没有跑通实测**：

```
权重 11.39 + fp32 动量 22.2                      = 33.6 GiB > 31.86  ⇒ PyT 真 AdamW 装不下（算术确定）
权重 11.39 + fp16 动量 11.1 + fp32 梯度 11.1     = 33.6 GiB > 31.86  ⇒ ORV 也装不下（实测 OOM）
```

也就是说：**在梯度也保持 fp32 的前提下，3B 全参 AdamW 在 32GB 上装不下，两家都装不下。**
要真正跑通，还差一步（梯度也压到 fp16 / 或分层更新后立即释放梯度）—— 这是下一阶段的工作。

**已经实测成立的显存结论**：小到中等规模模型（≤1B）在 ORV 上训练，显存占用明显低于 PyT
同等配置；以及上面那条"PyT 真 AdamW 在 3B 上必然装不下"的算术结论。

---

## 快速入门：从零到能训练，只要四步

一个完整可跑的脚本，`examples/beginner_pipeline.py`（**不需要 GPU 也能跑通**）：

```python
import tgai_go.train as T

# ① 导入语料（纯文本或 .jsonl 都行）
text = T.load_corpus("my_corpus.txt")

# ② 训练分词表（就在你的语料上训，不需要外部 tokenizer）
tok = T.train_tokenizer(text, vocab_size=4000)
ids = tok.encode(text)

# ③ 切块 + 建模型（这几个数字就是你要调的"超参"）
data  = T.tokens(ids, seq=128)
model = T.preset.transformer(vocab=tok.vocab_size, d_model=256, n_layers=6, n_heads=8)

# ④ 预训练
hist = T.fit(model, data, steps=2000, lr=3e-4)
print(hist["loss"][0], "->", hist["loss"][-1])
```

### 想换成自己的模型？

把第 ③ 步换成你自己写的 `nn.Module` 就行 —— `T.fit` 不关心模型长什么样，
只要它有 `parameters()` 和 `forward(x, y) -> loss`。

### 它真的能学会吗（实跑输出，纯 CPU，约 140 秒）

```
[1] 语料 7210 字符
[2] 分词表：Tokenizer(vocab_size=248, merges=211)
[3] token 数 769，切出 12 个训练块
[4] 模型参数量 147264，开始训练……
[fit] step 20/80 loss 5.212276
[fit] step 40/80 loss 4.589163
[fit] step 60/80 loss 4.038117
[fit] step 80/80 loss 3.425150

loss: 5.5404 -> 3.4252（降到原来的 62%）
✔ 训练链路是通的。
```

跑一次：`python examples/beginner_pipeline.py`（不给参数会自动生成示例语料；
也可以 `python examples/beginner_pipeline.py 你的语料.txt`）。

### 调参对照表（新手最常改的几个）

| 参数 | 典型值 | 说明 |
| --- | --- | --- |
| `vocab_size` | 2000~8000 | 太小词被切碎，太大在小语料上过拟合 |
| `seq` | 128~2048 | 越长越吃显存；受 `MAXLEN` 限制 |
| `d_model` / `n_heads` | 256/8、512/16、1024/16 | `d_model` 必须能被 `n_heads` 整除 |
| `n_layers` | 4~28 | 显存与时间的主要来源 |
| `lr` | 3e-4（预训练）、2e-5（微调） | 预训练大、微调小 |
| `steps` | 先 100 步看 loss 有没有降 | loss 不动先查 lr 和数据 |

### 想做 SFT / DPO / GRPO / PPO？

五种训练模式都有，都带对照测试：

| 模式 | 入口 |
| --- | --- |
| 预训练 | `T.fit(...)` |
| SFT | `T.SftDataset` + `T.fit(...)` |
| DPO | `T.DPOTrainer` |
| GRPO | `T.GRPOTrainer` |
| **PPO / RLHF** | `T.PPOTrainer` + `T.ppo.gae` / `T.ppo.ppo_loss` |

---

## 与 PyTorch 的 API 对齐程度

| 类别 | 已对齐 |
| --- | --- |
| 张量与 autograd | `Tensor`、广播、`backward()`、`requires_grad`、`no_grad` |
| `nn` | `Module`、`parameters()`、`state_dict`/`load_state_dict`、`zero_grad`、`Linear`、`Embedding`、`LayerNorm`、`RMSNorm`、`Conv1d/2d`（含分组）、`Dropout`、`BatchNorm1d/2d`、`MaxPool/AvgPool 1d/2d`、`Sequential`、`ModuleList` |
| 优化器 | `AdamW`、`SGD`、`Adafactor`、`LAMB`、`Lion`、`Muon` |
| 调度 | `CosineLR`、`LinearLR`、`StepLR`、`warmup` |
| 训练工具 | 梯度裁剪、梯度累积、`DataLoader`（`Dataset`/`IterableDataset`/`Sampler`/`collate`/多线程） |
| 精度 | `T.amp()`（fp16 / bf16，可编程开关，不用重启进程） |
| 生态互转 | `T.from_torch(...)`、`T.to_torch(...)`、checkpoint 读写 |
| 融合算子 | 101 个 CUDA 内核；`_cpu_only` 调用点 **0**（CUDA 上没有任何算子偷偷退回 CPU） |

**测试**：`553 passed`，覆盖上面每一行的数值对照。

---

## 安装

```bash
pip install tgai-orv
```

要求：Windows / Linux，NVIDIA 驱动（CUDA 12 运行时由驱动提供），纯 CPU 也能跑（慢）。

```python
import tgai_go.train as T
print(T.amp_state())        # 看 CUDA 是否可用
```

---

## 已知限制（写在明处）

1. **单卡**，无 DDP/FSDP。
2. **自定义 `forward` 不能自动迁移**（`from_torch` 会报错并列出支持的类型）。
3. **3B 全参 AdamW 在 32GB 上尚未跑通**（见上面 ⚠️ 那节），这是下一阶段工作。
4. **bf16** 的 API 已支持，但 V100 没有 bf16 硬件（会优雅回退 fp32）；A100/H100 的加速**我们无卡实测**。
5. V100 上 fp16 是 AMP 的唯一选择（无 bf16）。
6. 没有 `torch.compile`、没有模型动物园、没有多机。
7. 大于 0.5B 的热门模型微调**未实测**。

---

## 文档

- [与 PyTorch 的完整对照与差距](docs/REVIEW_2026-11-10_vs_PyTorch.md)
- [算子覆盖矩阵](docs/OP_COVERAGE.md)
- [C 端下沉清单 & 迁移边界](docs/C_END_AND_MIGRATION.md)
- [更新日志](CHANGELOG.md)

## License

MIT
