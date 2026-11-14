# TGAI ORV — Origin Requiem Vehicle（原生之初）

**用纯 Python 跑深度学习训练，不需要 PyTorch。**

自动微分、张量算子、优化器、训练循环都是自带的；CUDA 后端直接调 **驱动 + cuBLAS + NVRTC**，
所以 `pip install` 之后就能用，**不用装 CUDA Toolkit、不用编译任何 C++ 扩展**。

同一份代码也支持 CPU 与 Vulkan 后端（本次实测以 CUDA / V100 为准）。

---

## 安装

```bash
pip install tgai-orv
```

- Windows / Linux，Python 3.9+
- 有 NVIDIA 卡时：驱动需支持 CUDA 12（运行时由驱动提供，**不需要 Toolkit**）
- 纯 CPU 也能跑，只是慢

装完自检：

```python
import tgai_go.train as T
print(T.__version__)
print(T.amp_state())      # 看 CUDA 是否可用
```

---

## 30 秒上手

完整可运行脚本见 `examples/beginner_pipeline.py`（**不依赖 GPU**）：

```python
import tgai_go.train as T

# ① 导入语料（纯文本或 .jsonl）
text = T.load_corpus("my_corpus.txt")

# ② 在自有语料上训练分词表（不需要外部 tokenizer）
tok = T.train_tokenizer(text, vocab_size=4000)
ids = tok.encode(text)

# ③ 切块并建模型
data  = T.tokens(ids, seq=128)
model = T.preset.transformer(vocab=tok.vocab_size, d_model=256, n_layers=6, n_heads=8)

# ④ 预训练
hist = T.fit(model, data, steps=2000, lr=3e-4)
print(hist["loss"][0], "->", hist["loss"][-1])
```

---

## 执行档位：显存与速度的取舍

ORV 把「用多少显存换多少速度」做成**四组预设**，算子在运行时读策略决定行为，
所以**同一份模型代码**可以在四档之间切换，不需要改模型。

```python
import tgai_go.train as T

T.set_policy("speed")        # 切档：standard / speed / extreme / memory
print(T.get_policy())        # 看当前这一档的全部参数

T.set_policy_value("attn_tile", 1024)   # 单改某一项
print(T.policy("attn_tile"))            # 读某一项（带默认值：T.policy(k, 默认)）
```

### 四档是什么

| 档位 | 注意力实现 | 梯度检查点 | 定位 |
| --- | --- | --- | --- |
| `standard` | 物化（一次性算完 S×S） | 开 | 目前与 `extreme` 同参 |
| `speed` | 物化 | **关** | 用显存换速度 |
| `extreme` | 物化 | 开 | 目前与 `standard` 相同 |
| `memory` | 分块（flash 风格，不落 S²） | 开 | 用速度换显存 |

### 该选哪一档

| 你的处境 | 选 | 这一档实测是什么样 |
| --- | --- | --- |
| 显存够、想最快 | `speed` | 比 PyTorch 快约 2.1×，代价是约 1.8× 显存 |
| 想两头都不输 PyTorch | `extreme` | 显存约 0.80×、速度约 0.93×，两轴同时赢 |
| 显存吃紧 / 长上下文 | `memory` | 显存约 0.68×，且几乎不随上下文增长 |
| 不确定、先跑通再说 | `standard` | **目前与 `extreme` 是同一组参数**（物化注意力 + 检查点）， 也就是"两头都不输"那一档 |

> `standard` 与 `extreme` 现在参数完全相同。保留两个名字，是为了将来把 `standard` 调成更保守的默认
> （例如自动按可用显存选档），而不动已有用户写的 `extreme`。

> **注意**：`set_policy` 要在**建图之前**调用；中途切换只影响之后新建的算子。
> 显存与速度的完整测量见 [`docs/THREE_TARGETS.md`](docs/THREE_TARGETS.md)。

---

## 实测性能（V100 32GB）

模型：`seq=2048 / 8 层 / d=512 / 8 头 / top_k=2 / 共享专家`，**真实训练步**（前向+反向+AdamW）。
对照：同构 PyTorch = **4,633 MiB / 0.507 s**。

| 档位 | 峰值显存 | 步时 | 相对 PyTorch |
| --- | --- | --- | --- |
| `speed` | 8,329 MiB | **0.24 s** | 速度 **快 2.1×**，显存 1.8× |
| `extreme` | **3,692 MiB** | **0.47 s** | 显存 **0.80×**、速度 **0.93×** |
| `memory` | **3,169 MiB** | 0.81 s | 显存 **0.68×**、速度 1.6× |

### 上下文变长时

| seq | ORV `speed` | PyTorch | ORV `memory` |
| --- | --- | --- | --- |
| 512 | **0.21 s** | 0.286 s | — |
| 2048 | **0.40 s** → 0.24 s* | 0.360 s | 0.93 s → 0.81 s* |
| 8192 | 跑不了（需 ~80 GB） | 1.267 s | **3.30 s / 4,349 MiB** |

\* 两列是修复一处主机往返前后的对比，见 CHANGELOG。

- 短上下文（≤2048）：ORV 速度占优
- 长上下文（≥4096）：PyTorch 的融合注意力更快，但 ORV 的 `memory` 档显存几乎不涨

---

## 插件系统

算子、后端、损失函数、优化器、分词器、调度器 —— **六条扩展轴共用一个命名空间**，
一个文件就能同时适配 CPU / CUDA / Vulkan：

```python
from tgai_go.train import ops_plugin

@ops_plugin.register("my_op", backends=("cpu", "cuda", "vulkan"))
def my_op(x):
    ...
```

能力不足时**必须报错**而不是静默降级。细节见 `docs/`。

---

## 从 PyTorch 迁移

四步：换 import → 搬权重 → 改自定义 `forward` → 接训练循环。

```python
# import torch as T
import tgai_go.train as T
```

- 权重可用 `T.from_numpy(...)` / safetensors 直接搬，**不要重新训练**
- `Tensor` 语义与 PyTorch 接近，但**只支持 fp32**
- 不支持的能力会显式报错，不会静默算错

迁移检查清单见 [`docs/`](docs/)。

---

## 文档

| 文件 | 内容 |
| --- | --- |
| [`docs/THREE_TARGETS.md`](docs/THREE_TARGETS.md) | 三档目标的完整实测、上下文曲线、每次优化的前后对比 |
| [`docs/ORV_KERNEL_INVENTORY.md`](docs/ORV_KERNEL_INVENTORY.md) | 引擎里已有哪些内核/入口（避免重复造轮子） |
| `docs/OP_COVERAGE.md` | 算子支持矩阵 |
| [`CHANGELOG.md`](CHANGELOG.md) | 逐版本变更 |
| `bench/` | 逐模块微基准（CUDA event 计时），可单独运行 |

---

## 许可

MIT，见 [`LICENSE`](LICENSE)。
