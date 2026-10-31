# TGAI ORV — Origin Requiem Vehicle（原生之初）

<p align="center">
  <strong>训推一体，不依赖 PyTorch。· Train and infer with one codebase, no PyTorch.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.2.0-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.9+-green" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="license">
  <img src="https://img.shields.io/badge/self--test-12%2F12-brightgreen" alt="selftest">
  <img src="https://img.shields.io/badge/peak-1.43%C3%97%20vs%20PyTorch-red" alt="speed">
</p>

---

## ⚡ 先看这个 / Read This First

**同一份训练代码、同种子、同一个模型、各跑 600 步。**
Same training code, same seed, same model, 600 steps each. No cherry-picking.

| 显卡 GPU | 模型 | ORV | PyTorch | 结果 |
| --- | --- | --- | --- | --- |
| RTX 4090 | 10M · B78 S64 | **18.9 ms/step** | 27.1 ms | 🔥 **1.43× 更快** |
| Tesla V100 | 10M · B78 S64 | **41.4 ms/step** | 52.4 ms | **1.27× 更快** |
| NVIDIA A10 | 10M · B78 S64 | **49.3 ms/step** | 57.0 ms | **1.16× 更快** |
| NVIDIA A10 | 25.2M · D512 L6 FF2048 | **36.8 ms/step** | 39.8 ms | **1.08× 更快** |
| NVIDIA A10 | 25.2M · **bf16** | 30.0 ms/step | **22.3 ms** | 💀 **0.74×，我们输** |

> 最后一行红得刺眼，但我们不放小字：PyTorch 的 autocast 会缓存权重的 bf16 影子副本，
> 而我们是每次 GEMM 现 cast。**赢的地方写赢，输的地方写输。**
> The last row is a loss, stated plainly. PyTorch caches bf16 weight shadows; we cast per GEMM.

**我们不是来打酱油的，但也没赢下全部。剩下的，往下看。**

---

## 序章 · 何为 ORV / Prologue

> **Origin** —— 内核、autograd、优化器，一行一行从零写起。"原生"不是形容词，是事实陈述。
> **Requiem** —— 写给那个"想训模型就必须先装 PyTorch"的年代。挽歌不凄惨，是告别。
> **Vehicle** —— 载具。同一个模型、同一份量化、同一份词表，从训练场一路开到部署现场，中途不换船。

这不是又一个把 cuBLAS 包了一层的壳子。**三千多个 CUDA 算子内核在 NVRTC 里现编现用，autograd 自己推，AdamW 自己更。**

---

## 一句话 · 改掉那行 import

```python
# import torch as T                 # 旧时代的咒语 / the old incantation
import tgai_go.train as T           # 现在 / now — 其余代码一行不用改
```

`Tensor` / autograd / `optim.AdamW` / `Module` / `functional` —— **名字一样，语义一样，只是不再是 PyTorch。**

---

## 外来模型对齐 / Foreign Model Parity

我们不是只能跑自己写的结构。**真实开源权重 Qwen2.5-0.5B-instruct**（GQA 14:2、qkv bias、RoPE θ=1e6，
与我们的模型结构不同）跑在我们自己的张量算子上，与 HuggingFace `transformers` 逐位比对：

```
max|Δ| = 1.278e-04      相对误差 6.29e-06 ≤ 1e-5   ✅ PASS
我们:  [0.95818734, 6.6607723, 3.1660562, 1.2813374, 3.2732353]
参考:  [0.9581473,  6.66074,   3.1660404, 1.281328,  3.2732046]
```

复现（纯 CPU，不需要 GPU）：

```bash
python tools/qwen_ref.py        # 用 HF 生成参考 logits（一次性）
python tools/qwen2_parity.py    # 我们的实现 vs 参考，判据 ≤1e-5
```

**LLaMA 结构训练对齐**（RMSNorm + RoPE + SwiGLU + GQA，20 步）：
与 PyTorch 孪生逐步 loss 相对偏差 **1.59e-05**、训练后参数相对偏差 **4.90e-05**、loss 下降 99.1%。

**算子覆盖率**（以 PyTorch 公开面为基准，见 [docs/OP_COVERAGE.md](docs/OP_COVERAGE.md)）：
Tier-1 训练必需算子 **63 项，已具备 62 项（98%）**。

---

## 能替别人的模型训练吗？/ Can It Train Someone Else's Model?

我们不靠"我们的模型能跑"自证。下面是**外来结构 + 真实权重**的实测：

### ① 真实开源权重前向对齐（Qwen2.5-0.5B-instruct）

用 GGUF 里的真实权重，在我们引擎上跑前向，与 HuggingFace `transformers` 比对：

```
max|Δ| = 1.155e-04     相对误差 5.682e-06  ≤1e-5   ✅ PASS
```

处理了 Qwen2.5 与我们自有模型的全部差异：**GQA(14:2)**、**qkv bias**、
**独立 lm_head**（不绑词嵌入）、**RoPE θ=1e6**。

### ② 60M 级 LLaMA 结构在**我们的语料**上训通

```bash
python examples/llama_tiny.py --preset m60 --steps 20 \
    --corpus examples/_corpus_1m.txt --lr 3e-3
```

| 项 | 值 |
| --- | --- |
| 结构 | RMSNorm + RoPE + SwiGLU + GQA(8:2) + 无 bias，约 **50~56M 参数** |
| 语料 | 100 万字符真实中文 |
| loss | **8.217616 → 3.985692（降 51.5%）** |
| 与 PyTorch 孪生逐步相对偏差 | **1.16e-07** |
| 训练后参数相对偏差 | **2.60e-07** |

### ③ 真实 Qwen2.5 / Qwen3 前向对齐与微调

| 项 | 结果 |
| --- | --- |
| 前向对齐（官方 bf16 权重 vs transformers 现场算） | **相对误差 4.816e-06**（判据 ≤1e-5） |
| **过拟合单 batch（引擎正确性自检）** | loss **1.1433 → 0.000021**（4 条样本，25 步） |
| 小样本 SFT（36 条 + 4 留出，300 步） | 留出集 loss **0.965 → 0.335（−65.3%）** |
| 生成抽检（套训练模板，贪心） | 人格类提示 **2/3 通顺**；域外（数学）复读 |
| **Qwen3-0.6B 前向对齐**（真实权重切片 vs 官方 `Qwen3ForCausalLM`） | **相对误差 6.693e-07**（head_dim=128≠64 + q_norm + tie 全部生效） |
| **Qwen2.5-0.5B 微调**（真实权重，56+4 留出，60 步，补位已修正） | 留出集 loss **3.115 → 1.558（−50.0%）** |
| **Qwen3-0.6B 微调**（真实权重，56+4 留出，60 步，补位已修正） | 留出集 loss **3.204 → 1.500（−53.2%）** |

> **诚实边界**：`过拟合单 batch` 证明的是**引擎的梯度/掩码/优化器/数据链路都正确**；
> 而"训出可用模型"**还没有证明** —— 2000 条样本的多组实验里，留出集改善只有 −0.1%~−1.5%，
> 域外输入会复读。全部实验（含失败）的命令、超参、日志、判据都在
> **[docs/FINETUNE_RESULTS.md](docs/FINETUNE_RESULTS.md)**。

### ④ SFT 链路（CPU 冒烟）

`examples/qwen25_sft.py` 支持 GGUF / safetensors 两种权重、AdamW、固定探针评估。
CPU 冒烟（随机迷你权重）：探针 loss **5.154 → 4.662** 单调下降。

> 这三组证据都能在**纯 CPU** 上复现（`pytest tests` 里有对应的 CI 用例）。
> GPU 上的真实 0.5B 微调还需要跑一次，结果会补到这里。

---

## 模型转换 / Convert

训练或部署别人的模型，第一步是把权重换成我们的格式：

```bash
python tools/gguf_to_tg.py --input qwen2.5-0.5b-fp16.gguf --output qwen25.TG --dtype fp16
python tools/verify_gguf_tg.py --gguf qwen2.5-0.5b-fp16.gguf --tg qwen25.TG   # 转完就验
```

实测 Qwen2.5-0.5B：**16.6 秒**、944 MB（FP16）、**290/290 张量逐位一致（max|Δ|=0）**。
支持 `llama / mistral / qwen2 / qwen3 / gemma2 / phi3`；细节见 [docs/GGUF_TO_TG.md](docs/GGUF_TO_TG.md)。

---

## 一切皆插件 / Everything Is a Plugin

核心只保留 **LLM 训练必需**的算子；图形学、科研、私有算子**不进核心**，走插件。
这样核心永远精简可自检，生态由插件自由生长。

```python
from tgai_go.train import ops_plugin

# 组合式：只用已有算子，反向自动生效
ops_plugin.register("softplus", lambda x: (x.exp() + 1.0).log(), kind="tensor")
x.softplus()          # 立即可用，梯度也是对的
```

```python
# 自带反向：numpy 前向 + 解析反向（精确 GELU 需要 erf，就不该进核心）
ops_plugin.register_numpy("gelu_exact", fwd, bwd, kind="tensor")
```

**实测**（`examples/plugins/`，与 torch 对照）：

| 插件 | 前向 | 反向 |
| --- | --- | --- |
| `softplus`（组合式，一行反向都没写） | 2.4e-07 | **6.0e-08** |
| `gelu_exact`（自带解析反向） | 2.4e-07 | **1.2e-07** |

发现方式：手工 import / `TG_OPS_PATH` 目录 / setuptools 入口点（`pip install` 后自动）。
细节见 [docs/PLUGINS.md](docs/PLUGINS.md)。

---

## 装 / Install

```bash
pip install tgai-orv              # 机器上已有 CUDA Toolkit 或 torch 的 CUDA 库
pip install "tgai-orv[cuda]"      # 裸机器：把 cuBLAS / NVRTC / CUDA runtime 一起装进 venv
```

```bash
python -m tgai_go.train.doctor    # 12 项自检，失败会写清哪个算子/什么形状/怎么修
```

**已验证**：全新的、**没有 torch** 的 venv，不设任何环境变量，`pip install tgai-orv` 后
`doctor` **12/12 PASS**。引擎会自动去本机任何 Python 环境里找 `nvidia/*/lib`；
万一找不到，报错会直接告诉你 `pip install "tgai-orv[cuda]"` 或设 `TGAI_CUDA_BIN`（支持多路径，`:` 分隔）。

> 发布前自检：`python tools/check_package.py --build` —— 检查 wheel 里文件是否齐、
> `.cu` 是否随包、装进干净 venv 后能否 import（今天两次「装上才发现缺件」都是它挡住的）。
>
> PyPI: https://pypi.org/project/tgai-orv/ · 当前 `0.2.0`

---

## 实力清单 / The Arsenal

- 🗡️ **真·融合核**：RMSNorm · RoPE · SwiGLU · Attention（含 flash 路径）· MoE，外加 CUDA Graph。
- 🧪 **12 项自检**：`python -m tgai_go.train.doctor`。崩了会告诉你**哪个算子、什么形状、怎么修** —— 不许裸报错。
- 📉 **1 GB 语料只吃 600 MiB 内存**：流式管线。旧实现要 17.5 GiB，相当于"一个语料吃掉十六倍内存"。
- ♻️ **续训无损**：40+40 步 vs 连续 80 步，末 loss 差 **0.06%**。断电重启不心疼。
- 📐 **14 组形状实测通过**：`dh=8~128`、`S=8~256`、`FF=1000`（**非 128 倍数**）、`L=16`。
- ⚡ **混合精度**：`TG_BF16=1` / `TG_FP16=1`；A10 上 bf16 跑 600 步 loss 偏差 **0.03%**。
- 🔗 **数据并行**：单卡起 2 个 rank，最终参数 **SHA256 完全一致** —— 数学上是对的。
- 🚀 **一条命令起步**：`bash scripts/quickstart.sh`，自动查环境、装依赖、跑自检、告诉你下一步。

---

## 拔剑 / Quick Start

```bash
git clone https://github.com/JXW666NB/TGAI-ORV.git && cd TGAI-ORV
bash scripts/quickstart.sh          # Windows: scripts\quickstart.bat
```

想立刻见血 / train something right now：

```bash
PYTHONPATH=$PWD/python python3 examples/exam2_chat10m_tgai.py --device cuda --steps 600 --out chat10m
```

---

## 它不是什么 / What It's NOT（诚实的部分）

**吹牛要有底线。** 下面这张表里，🟡 那行我们没有藏。

| 关卡 Gate | 状态 |
| --- | --- |
| L0 · 数值正确 | ✅ 前向/反向 ≤1e-5，600 步 loss 偏差 ≤2%，无 NaN |
| L1 · 能训 | ✅ 形状 / 1 GB 语料 / 速度 / 显存 / 续训 **全部通过** |
| L2 · 能被人用 | ✅ 一条自检、一页文档、报错带修法、后端一行切换 |
| L3 · 生产级 | 🟡 **24h 长稳与 2 卡加速比缺硬件未验证** |

⇒ 所以我们给自己的标签是 **「可用（自用 / 小团队）」**，**不是"生产级"**。
⇒ Label: **"usable (self / small team)"** — not "production-ready".

每一个数字都在真机上测过，证据与判据全部公开：[`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md)

---

## 兵器谱 / Layout

```
TGAI-ORV/
├── python/tgai_go/train/     训练引擎（运行时 + autograd + 优化器）
├── training/kernels/         CUDA 算子源码（.cu，NVRTC 运行时编译）
├── examples/                 可跑示例（10M 对话模型 + 流式数据）
├── tools/                    accept / soak / profile / sync
├── scripts/                  quickstart.sh · quickstart.bat
├── docs/                     QUICKSTART_TRAIN.md · PRODUCTION_READINESS.md
├── src/ cuda/ cpu/ embedded/ shaders/ tcc/    推理引擎
└── third_party/              上游依赖
```

## 同门 / Sibling

| Repo | 定位 |
| --- | --- |
| [**TGAI GO**](https://github.com/JXW666NB/TGAI-GO) | 纯推理引擎。跨平台，从 ESP32 的 8 MB PSRAM 到 CUDA 服务器 |
| **TGAI ORV**（本仓库） | 完全体。推理 + 训练，训完直接部署 |

## License

[MIT](LICENSE) © 2026 JXW666NB —— 拿走随便用，留着这行字就行。
