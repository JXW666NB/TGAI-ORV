# TGAI ORV — Origin Requiem Vehicle（原生之初）

<p align="center">
  <strong>用 PyTorch 的写法，在不装 PyTorch 的机器上训练模型。</strong><br>
  <em>Write like PyTorch. Train without it.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.9.28-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.9+-green" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="license">
  <img src="https://img.shields.io/badge/tests-558%20passed-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/CUDA%20kernels-101-blueviolet" alt="kernels">
</p>

---

# 第一部分：这是什么

## 一句话

**ORV 是一个训练深度学习模型的 Python 库，它不依赖 PyTorch。**

```bash
pip install tgai-orv
```

安装完成后，机器上不需要存在 `torch`，也能训练模型。

## 展开说

训练一个神经网络通常需要三样东西：

1. **张量计算库** —— 提供矩阵乘法、卷积等运算（PyTorch 用 `torch` + cuBLAS 做这件事）
2. **自动求导** —— 反向传播时自动算出每个参数的梯度（PyTorch 的 `autograd`）
3. **优化器** —— 按梯度更新参数（PyTorch 的 `optim.AdamW` 等）

PyTorch 把这三样打成一个包。**ORV 把这三样全部自己实现了一遍**：

| 组成部分 | ORV 的实现方式 |
| --- | --- |
| 张量与算子 | 101 个 CUDA 内核，用 NVRTC 在运行时编译；矩阵乘走 cuBLAS |
| 自动求导 | 纯 Python 实现的动态计算图 |
| 优化器 | 自研 AdamW / SGD / Adafactor / LAMB / Lion / Muon |
| 神经网络层 | 自研 `nn.Linear` / `nn.LayerNorm` / `nn.Conv2d` 等 |

因此 ORV 不是"PyTorch 的封装"或"另一个前端" —— 它**替换**了 PyTorch 本身。

## 代码长什么样

ORV 刻意保持与 PyTorch 相同的 API，所以代码写起来几乎一样：

```python
import tgai_go.train as T          # 需要 PyTorch 时这里写 import torch as T

model = T.nn.Sequential(
    T.nn.Linear(784, 256),
    T.nn.ReLU(),
    T.nn.Linear(256, 10),
)
opt  = T.optim.AdamW(model.parameters(), lr=1e-3)
loss = T.functional.cross_entropy(model(x), y)
loss.backward()
opt.step()
```

上面这段代码在 PyTorch 上能跑，在 ORV 上也能跑，只是 `import` 那一行不同。

## 适用范围

| 适合 | 不适合 |
| --- | --- |
| 单张 NVIDIA GPU 上的训练 | 多卡 / 多机训练（无 DDP / FSDP） |
| 自己写的模型结构 | 直接调用 `torchvision` 等生态包 |
| 显存紧张的场景（3B 级模型跑在 32GB 卡上） | 需要 `torch.compile` 的极致吞吐场景 |
| 想摆脱 PyTorch 依赖的部署环境 | 需要 CUDA 之外的加速后端（Vulkan 正在开发） |

---

# 第二部分：优势，以及几个常见疑问

## 核心优势：在显存受限的单卡上把训练跑完

ORV 最直接的用途是**省显存**。一个具体的实测结果：

> V2.7 模型（28 层、2.97B 参数、MoE 结构）可以在 **V100 32GB** 上做全参训练，
> 峰值显存 **18.93 GiB**，**不需要**把参数或优化器状态换出到主机内存。

对比 PyTorch：同样的模型用标准 AdamW，仅优化器状态就需要 22.2 GiB，
加上 11.39 GiB 权重就已经超过 32GB —— 还没开始算激活和梯度。

省显存的三个手段，**都不降低计算精度**：

| 手段 | 省下什么 | 代价 |
| --- | --- | --- |
| 逐层重算（梯度检查点） | 不同时保存 28 层激活，只留一层 | 反向时多算一遍前向 |
| 单份优化器状态（Muon） | 2D 参数只需一份动量；AdamW 需要两份 | Muon 专用于 2D 参数，1D 参数仍走 AdamW |
| 逐层即时更新 | 梯度用完立刻释放，不同时驻留 28 层 | 优化器调用次数从 1 次变成 28 次 |

**计算本身始终是 fp32**：参数、激活、梯度都是 fp32，AMP 只是把矩阵乘的输入转成 fp16
再用 fp32 累加。所以不存在"用低精度换显存"。

## 常见疑问

### 数学对齐了吗？

同一模型、同一份权重、同一种子，ORV 与 PyTorch 各训练 8 步并逐步比较 loss：

```
最大相对差 = 1.44e-07        （判据 1e-4）
```

逐算子对照（ORV 与 torch 同类实现，均为实测值）：

| 算子 / 路径 | 相对差 |
| --- | --- |
| gather / scatter、CUDA Graph 重放的梯度、融合 fp16-AdamW | **0.00e+00**（逐位相同） |
| `masked_fill`、`repeat_interleave` | 0.00e+00 |
| LayerNorm | 4.38e-08 |
| `sqrt` / `log` / `gelu` | 5.92e-08 / 7.50e-08 / 7.78e-08 |
| `tanh` / `std` | 2.39e-07 / 2.02e-08 |
| Conv2d / 分组卷积 | 2.39e-07 |
| `CausalSelfAttention`（前向与反向） | 5.02e-08 |
| LR 调度（cosine/linear/warmup） | 4.55e-15 |
| **fp16 GEMM（AMP）** | 2.78e-04 ~ 2.88e-04（与 torch AMP 同量级，属 fp16 自身精度） |

**外部真实权重**：Qwen2.5-0.5B-instruct 在 ORV 上运行，与 HuggingFace 输出相对差 **6.29e-06**（判据 1e-5）。

### 能微调热门模型吗？

已验证至 0.5B 规模：

```python
import tgai_go.train as T

model = T.from_torch(hf_model)          # 逐位复制权重，支持 Linear/Embedding/LayerNorm/
                                        # RMSNorm/Conv1d/2d/Dropout/BatchNorm/池化/Sequential
hist  = T.fit(model, data, steps=2000, lr=2e-5)
```

已验证对象：**Qwen2.5-0.5B-instruct**（GQA 14:2 + qkv bias + RoPE θ=1e6）与 HF 逐位对齐；
LLaMA 结构（RMSNorm+RoPE+SwiGLU+GQA）20 步训练与 PyTorch 孪生 loss 相对偏差 **1.59e-05**。

**能力边界**：7B 以上的热门模型尚未做显存实测，0.5B 是已验证的上限。

### 能训练自己的模型吗？

可以。任意 `nn.Module` 都能训练，`T.fit` 只要求 `parameters()` 与 `forward(x, y) -> loss`。
从语料到训练的完整流程见第三部分。

### 速度如何？

**V100 32GB，seq=512，逐层反向：**

| 层数 | ORV | PyTorch | 结论 |
| --- | --- | --- | --- |
| 4 层 | **78.1 ms** | 135.6 ms | ORV 快 **1.74×** |
| 8 层 | **187.5 ms** | 264.1 ms | ORV 快 **1.41×** |
| 14 层 | **327.6 ms** | 484.9 ms | ORV 快 **1.48×** |
| 28 层（真实 V2.7） | ~2038 ms | 2056 ms | 基本持平 |

**大矩阵乘（2048×2048×6144）：**

| 口径 | 耗时 |
| --- | --- |
| ORV fp32 | 4.33 ms |
| **ORV AMP（fp16）** | **0.67 ms（6.45×）** |
| PyTorch AMP | 0.71 ms |

**ORV 自身加速（同一份代码，优化前后）：**

| 场景 | 优化前 | 优化后 | 倍数 |
| --- | --- | --- | --- |
| T=128 整步 | 1843 ms | **721 ms** | 2.56× |
| T=512 整步（28 层） | 2038 ms | **754 ms** | 2.70× |
| 单层注意力前向（seq=256） | 9.8 ms | **2.7 ms** | 3.66× |

提速来自逐层 CUDA Graph、注意力堆叠实现（消除 16 次小 kernel）、融合 fp16-AdamW。

### 显存实测（3B MoE 全参训练，V100 32GB）

| 配置 | 优化器状态 | 峰值显存 | 整步耗时 | 结果 |
| --- | --- | --- | --- | --- |
| AdamW，fp32 动量 | 22.1 GiB | ≥46 GiB 需求 | — | 不可行 |
| AdamW，fp16 动量，全部梯度驻留 | 11.1 GiB | 33.6 GiB 需求 | — | **实测 OOM** |
| **AdamW，fp16 动量 + 逐层即时更新** | 11.1 GiB | **24.78 GiB** | **709 ms** | ✅ **推荐** |
| **Muon，fp16 动量 + 逐层即时更新** | 5.5 GiB | **18.93 GiB** | 62.9 s | ⚠️ 省显存但**极慢** |

**结论与取舍**：

- **AdamW 是 3B 上的默认选择**：峰值 24.78 GiB，整步 709 ms，比 PyTorch 的 Adafactor 步（911 ms）还快。
- **Muon 只在显存极度紧张时用**：峰值低 5.9 GiB，但 Newton-Schulz 要对 2550 个张量逐个做，
  实测 62.9 秒/步，慢 89 倍。这是 Muon 的固有代价，不是实现缺陷 —— 用它必须接受这个速度。

> 关于 Muon：它对 2D 参数做 Newton-Schulz 正交化（5 次迭代矩阵乘）。
> 在小模型上开销可忽略，但 3B 有 2550 个参数张量，逐张量做正交化就是不可接受的。
> 若要用 Muon 训练大模型，需要把"按层合并成正交化组"或换更低迭代次数 —— 属于后续工作。

### 降低精度了吗？

`fp16` 只用于两处：优化器动量的存储、矩阵乘的输入。前者实测影响：

| 指标 | 结果 |
| --- | --- |
| 参数偏差 | 4.6e-05 |
| 更新方向余弦相似度 | **1.000000** |
| loss 偏差 | 3.2e-05 |

若希望完全不用 fp16，可用 `Muon(fp16_mom=False)` 走 fp32 动量（3B 需要约 11.1 GiB 优化器状态）。

---

# 第三部分：新手入门

## 四步完成从语料到训练

完整可运行脚本见 `examples/beginner_pipeline.py`（**不依赖 GPU**）：

```python
import tgai_go.train as T

# ① 导入语料（纯文本或 .jsonl）
text = T.load_corpus("my_corpus.txt")

# ② 在自有语料上训练分词表（无需外部 tokenizer）
tok = T.train_tokenizer(text, vocab_size=4000)
ids = tok.encode(text)

# ③ 切块并构建模型（以下参数即需要调整的超参）
data  = T.tokens(ids, seq=128)
model = T.preset.transformer(vocab=tok.vocab_size, d_model=256, n_layers=6, n_heads=8)

# ④ 预训练
hist = T.fit(model, data, steps=2000, lr=3e-4)
print(hist["loss"][0], "->", hist["loss"][-1])
```

### 实际运行输出（纯 CPU，约 140 秒）

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

运行方式：`python examples/beginner_pipeline.py`（无参数时自动生成示例语料；
也可 `python examples/beginner_pipeline.py corpus.txt`）。

## 替换为自定义模型

将第 ③ 步替换为任意 `nn.Module` 即可 —— `T.fit` 不依赖模型结构。

## 超参参考

| 参数 | 典型值 | 说明 |
| --- | --- | --- |
| `vocab_size` | 2000~8000 | 过小导致切词过碎，过大在小语料上易过拟合 |
| `seq` | 128~2048 | 越长越占显存 |
| `d_model` / `n_heads` | 256/8、512/16、1024/16 | `d_model` 必须能被 `n_heads` 整除 |
| `n_layers` | 4~28 | 显存与耗时的主要来源 |
| `lr` | 3e-4（预训练）、2e-5（微调） | 预训练较大、微调较小 |
| `steps` | 先跑 100 步观察 loss | loss 不动时先检查 lr 与数据质量 |

## 五种训练模式

| 模式 | 入口 |
| --- | --- |
| 预训练 | `T.fit(...)` |
| SFT | `T.SftDataset` + `T.fit(...)` |
| DPO | `T.DPOTrainer` |
| GRPO | `T.GRPOTrainer` |
| PPO / RLHF | `T.PPOTrainer` + `T.ppo.gae` / `T.ppo.ppo_loss` |

---

# 附录：迁移与 API 对齐

## 替换一行 import

```python
# import torch as T
import tgai_go.train as T
```

`Tensor` / autograd / `nn.Module` / `nn.Linear` / `optim.AdamW` / `functional` / `DataLoader` /
`state_dict` —— 名称与语义保持一致。

## 与 PyTorch 的 8 项真实差异

| # | 差异 | 影响与处理方式 |
| --- | --- | --- |
| 1 | **仅支持单卡**，无 DDP/FSDP | 多卡训练不在范围内 |
| 2 | 自定义 `nn.Module` 子类的 `forward` **无法自动迁移** | 权重可用 `from_torch` 搬运，`forward` 需用 ORV 的层重写（通常 10~40 行） |
| 3 | `Tensor.max(dim=)` 只返回值，**不返回索引元组** | 需要索引时配合 `argmax` |
| 4 | CUDA 上 `permute/reshape` 是**真拷贝**（PyTorch 为视图） | 大张量频繁 permute 会增加显存占用 |
| 5 | `Tensor.to(device)` / `.cuda()` / `.cpu()` 是**叶子级搬运，会断开 autograd 链** | 仅用于数据搬运，不可放在 forward 中间 |
| 6 | 无 `torch.compile` 对应物 | 加速依靠 `T.amp()` 与逐层 CUDA Graph |
| 7 | AMP 语义不同：参数/激活/梯度**全程 fp32**，仅 GEMM 输入 cast | **因此不需要 GradScaler**（保留该接口只为兼容迁移代码） |
| 8 | 无 `torchvision` / `torchaudio` 等生态包 | 只做训练核心 |

## API 对齐清单

| 类别 | 已对齐 |
| --- | --- |
| 张量与 autograd | `Tensor`、广播、`backward()`、`requires_grad`、`no_grad` |
| `nn` | `Module`、`parameters()`、`state_dict`/`load_state_dict`、`zero_grad`、`Linear`、`Embedding`、`LayerNorm`、`RMSNorm`、`Conv1d/2d`（含分组）、`Dropout`、`BatchNorm1d/2d`、`MaxPool/AvgPool 1d/2d`、`Sequential`、`ModuleList` |
| 优化器 | `AdamW`、`SGD`、`Adafactor`、`LAMB`、`Lion`、`Muon` |
| 调度 | `CosineLR`、`LinearLR`、`StepLR`、`warmup` |
| 训练工具 | 梯度裁剪、梯度累积、`DataLoader`（`Dataset`/`IterableDataset`/`Sampler`/`collate`/多线程） |
| 精度 | `T.amp()`（fp16 / bf16，可编程开关，无需重启进程） |
| 生态互转 | `T.from_torch(...)`、`T.to_torch(...)`、checkpoint 读写 |
| 融合算子 | 101 个 CUDA 内核；CUDA 上 CPU-only 算子调用点 **0** |

**测试**：`558 passed`，覆盖上表每一行的数值对照。

---

## 安装

```bash
pip install tgai-orv
```

要求：Windows / Linux，NVIDIA 驱动（CUDA 12 运行时由驱动提供）；纯 CPU 亦可运行（较慢）。

```python
import tgai_go.train as T
print(T.amp_state())        # 检查 CUDA 是否可用
```

## 已知限制

1. **仅支持单卡**，无 DDP/FSDP。
2. **自定义 `forward` 无法自动迁移**（`from_torch` 会报错并列出支持的类型）。
3. **bf16** 的 API 已支持，但 V100 无 bf16 硬件（会优雅回退 fp32）；A100/H100 的加速尚无硬件实测。
4. 无 `torch.compile`、无模型动物园、无多机支持。
5. 大于 0.5B 的热门模型微调尚未实测。
6. 纯 Python 实现的 BPE 分词器仅保证可用与可复现，不追求大语料吞吐；生产环境建议替换为成熟分词器。

## 文档

- [与 PyTorch 的完整对照与差距](docs/REVIEW_2026-11-10_vs_PyTorch.md)
- [算子覆盖矩阵](docs/OP_COVERAGE.md)
- [C 端下沉清单 & 迁移边界](docs/C_END_AND_MIGRATION.md)
- [更新日志](CHANGELOG.md)

## License

MIT
