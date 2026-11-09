# QUICKSTART：用 ORV 训练你自己的模型（不需要 `.tg`）

`tgai-orv` 是**通用训练库**：TGAI 的 `.tg` 只是样例，你可以用公开 API 定义自己的模型并训练。
本文件给最小可跑示例与已验证的能力边界。

## 1. 安装

```bash
pip install tgai-orv
# 机器上没有 CUDA Toolkit / torch 时，把 CUDA 运行库一起装进 venv：
pip install "tgai-orv[cuda]"
```

## 2. 定义模型

```python
import numpy as np
import tgai_go.train as T

class TinyTransformer(T.nn.Module):
    def __init__(self, vocab=32, dim=16, heads=4, layers=2):
        super().__init__()
        self.vocab, self.dim, self.heads, self.layers = vocab, dim, heads, layers
        self.emb = T.nn.Embedding(vocab, dim)
        self.ln1 = T.nn.ModuleList([T.nn.RMSNorm(dim) for _ in range(layers)])
        self.ln2 = T.nn.ModuleList([T.nn.RMSNorm(dim) for _ in range(layers)])
        self.wq = T.nn.ModuleList([T.nn.Linear(dim, dim) for _ in range(layers)])
        self.wk = T.nn.ModuleList([T.nn.Linear(dim, dim) for _ in range(layers)])
        self.wv = T.nn.ModuleList([T.nn.Linear(dim, dim) for _ in range(layers)])
        self.wo = T.nn.ModuleList([T.nn.Linear(dim, dim) for _ in range(layers)])
        self.w1 = T.nn.ModuleList([T.nn.Linear(dim, 4 * dim) for _ in range(layers)])
        self.w2 = T.nn.ModuleList([T.nn.Linear(4 * dim, dim) for _ in range(layers)])
        self.fn = T.nn.RMSNorm(dim)
        self.head = T.nn.Linear(dim, vocab)

    def forward(self, ids):
        n = len(ids)
        mask = T.tensor(np.triu(np.full((n, n), -1e9, np.float32), k=1))
        x = self.emb(T.tensor(np.asarray(ids, np.int64)))
        for i in range(self.layers):
            h = self.ln1[i](x)
            q, k, v = self.wq[i](h), self.wk[i](h), self.wv[i](h)
            sc = (q @ k.T) * (1.0 / np.sqrt(self.dim / self.heads))
            x = x + self.wo[i](T.nn.functional.softmax(sc + mask, axis=-1) @ v)
            h2 = self.ln2[i](x)
            x = x + self.w2[i](T.nn.functional.silu(self.w1[i](h2)))
        return self.head(self.fn(x))
```

## 3. 训练（AdamW + 余弦调度 + 梯度累加 + 裁剪）

```python
model = TinyTransformer()
params = list(model.parameters())

opt   = T.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
sched = T.optim.lr_scheduler.CosineLR(opt, T_max=1000, eta_min=1e-4, warmup_steps=50)
acc   = T.optim.Accumulator(params, steps=4)          # 梯度累加 4 个微批

for step, batch in enumerate(loader):
    loss = model(batch.input_ids).pipe(...)           # 你的损失
    loss.backward()
    if not acc.tick():
        continue
    acc.finalize()                                    # 按 1/steps 平均
    T.nn.utils.clip_grad_norm_(params, 1.0)           # 裁剪在累加之后
    sched.step()                                      # 写回 opt.lr
    opt.step()
    opt.zero_grad()
    acc.reset()
```

完整可跑版本：`examples/custom_model_quickstart.py`（纯 CPU、秒级、随机合成数据）。

## 4. 与 PyTorch 的一致性（实测）

同一个模型（同初始化、同结构、同超参）分别用 ORV 与 torch 训练 8 步：

```
逐 step loss 最差相对误差 = 1.44e-07     （判据 1e-4）
```

测试：`tests/test_custom_model_parity.py`。其余逐项对照（梯度、优化器、调度器、裁剪、累加、
DPO、GRPO）见 `docs/TRAIN_V27_ORV.md` 的 §30–§32。

## 5. 能力边界（如实说明）

**已支持**
- 反向传播：本库自己的 autograd（`Tensor` 运算、`backward()`），算子覆盖本项目需要的部分
- 优化器：AdamW / AdamW(fp16 动量) / Int8AdamW / Muon / SGD / Lion / Adafactor
- 调度器：Constant / Linear / Cosine(+warmup) / Step / Lambda
- 梯度累加、梯度裁剪、激活重算（`T.checkpoint`）
- 训练形态：预训练 / SFT；DPO；GRPO（组内优势 + 裁剪 + KL）
- 单卡 32 GB 内完成 3B 全参 AdamW 训练（torch 在同卡需要 32.6 GB，实测 OOM）

**未支持 / 限制**
- **多卡**（DDP/FSDP）：不在当前结论内
- **通用算子覆盖**：没有 conv、没有任意 `gather`/高级索引；自定义层只能用已有算子拼
- `token_logprobs`（GRPO）需要一份 `[T,V]` one-hot，**词表大时显存开销明显**
- DPO 的 `seq_logprob` 按序列逐个调用（batch 内多几个 kernel）
- 没有内置的 `DataLoader`/`Dataset` 抽象（示例里直接用 numpy 数组）
- 与 `torch.nn.Module` 不互相兼容：不能把 torch 层直接塞进来
