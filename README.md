# TGAI ORV — Origin Requiem Vehicle（原生之初）

[![PyPI](https://img.shields.io/pypi/v/tgai-orv.svg)](https://pypi.org/project/tgai-orv/)
[![GitHub release](https://img.shields.io/github/v/release/JXW666NB/TGAI-ORV)](https://github.com/JXW666NB/TGAI-ORV/releases)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/tgai-orv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/JXW666NB/TGAI-ORV.svg)](https://github.com/JXW666NB/TGAI-ORV/stargazers)

**当前版本：1.0.32**（2026-11-25 发布 · [PyPI](https://pypi.org/project/tgai-orv/) · [Release 说明](https://github.com/JXW666NB/TGAI-ORV/releases/tag/v1.0.32)）

**用纯 Python 跑深度学习训练，不需要 PyTorch。** 自动微分、张量算子、优化器、训练循环都是自带的；
CUDA 后端直接调**驱动 + cuBLAS + NVRTC**，所以 `pip install` 之后就能用，**不用装 CUDA Toolkit、
不用编译任何 C++ 扩展**。同一份代码也支持 CPU 与 Vulkan 后端。

**它不是什么**：不是 PyTorch 的完整替代 —— 只覆盖训练所需的能力子集（张量语义以 fp32 为主，
缺的能力会显式报错，不静默降级）；也不是"只能训练"的框架 —— 训练产物 `.TG` 由同包内的推理引擎直接加载（推训一体）。

---

## 特性一览

| 特性 | 说明 | 证据 |
| --- | --- | --- |
| 纯 Python 训练栈 | autograd / 算子 / 优化器 / 训练循环全自带，不依赖 PyTorch | [`docs/QUICKSTART_TRAIN.md`](docs/QUICKSTART_TRAIN.md) |
| 四档执行策略 | `standard` / `speed` / `extreme` / `memory`，运行时切换显存⇄速度 | [`docs/THREE_TARGETS.md`](docs/THREE_TARGETS.md) |
| 长上下文 64K | CSA 全局分支 query 分块 + 反向重算，显存近似线性 | [`docs/LONG_CONTEXT_RESULTS.md`](docs/LONG_CONTEXT_RESULTS.md) |
| 推训一体 | 训练产物 `.TG` 直接推理，不用转 GGUF / ONNX | [`docs/INFERENCE_QUICKSTART.md`](docs/INFERENCE_QUICKSTART.md) |
| 插件系统 | 算子 / 后端 / 损失 / 优化器 / 分词器 / 调度器六轴 + `.tgop` 整包 | [`docs/PLUGIN_SPEC.md`](docs/PLUGIN_SPEC.md) |
| 自省与离线 Wiki | `T.help` 运行时自省，单文件离线 Wiki、零外链 | [`docs/HELP_SYSTEM.md`](docs/HELP_SYSTEM.md) |
| 数据并行 | 每卡完整模型 + 共享内存归约，不依赖 NCCL / MPI | [`docs/DP_2GPU_RUNBOOK.md`](docs/DP_2GPU_RUNBOOK.md) |
| PyTorch 迁移 | 权重直接搬，逐算子对齐，缺的能力显式报错 | [`docs/C_END_AND_MIGRATION.md`](docs/C_END_AND_MIGRATION.md) |

---

## 安装

```bash
pip install tgai-orv
```

- Windows / Linux，Python 3.9+
- 有 NVIDIA 卡时：驱动需支持 CUDA 12（运行时由驱动提供，**不需要 Toolkit**）
- 纯 CPU 也能跑，只是慢；Linux 裸机（无 CUDA runtime、无 torch）可改用 `pip install "tgai-orv[cuda]"`，
  把 cuBLAS / NVRTC / runtime 装进同一 venv（见 [`docs/QUICKSTART_CUSTOM_MODEL.md`](docs/QUICKSTART_CUSTOM_MODEL.md)）

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

## 核心概念

### 执行档位：显存与速度的取舍

ORV 把「用多少显存换多少速度」做成**四组预设**，算子在运行时读策略决定行为，所以**同一份模型代码**
可以在四档之间切换，不需要改模型。

```python
import tgai_go.train as T

T.set_policy("speed")        # 切档：standard / speed / extreme / memory
print(T.get_policy())        # 看当前这一档的全部参数

T.set_policy_value("attn_tile", 1024)   # 单改某一项
print(T.policy("attn_tile"))            # 读某一项（带默认值：T.policy(k, 默认)）
```

| 档位 | 注意力实现 | 梯度检查点 | 定位 |
| --- | --- | --- | --- |
| `standard` | 物化（一次性算完 S×S） | 开 | 目前与 `extreme` 同参 |
| `speed` | 物化 | **关** | 用显存换速度 |
| `extreme` | 物化 | 开 | 目前与 `standard` 相同 |
| `memory` | 分块（flash 风格，不落 S²） | 开 | 用速度换显存 |

怎么选：显存够、想最快 → `speed`；想两头折中 → `extreme`（或等价的 `standard`）；显存紧 / 长上下文 →
`memory`；不确定先跑通 → `standard`。`standard` 与 `extreme` **现在参数完全相同**，保留两个名字是为了将来
把 `standard` 调成更保守的默认而不影响已有代码。

> `set_policy` 要在**建图之前**调用；中途切换只影响之后新建的算子。
> 档位间的**定量倍率**来自历史测量、当前版本待复测，完整表见 [`docs/BENCHMARKS_HISTORY.md`](docs/BENCHMARKS_HISTORY.md)。

### 长上下文：CSA 的 query 分块

CSA 全局分支默认 `cq=2048`：**seq ≤ 2048 自动走原路径、逐位不变**；更长序列自动 query 分块 + 反向重算，
显存从 O(seq²) 降到近似线性（实测 64K 见下方性能摘要）。要完全回退设 `TG_CSA_ATTN_CQ=0`。

---

## 性能摘要

> **口径约定**：每个表都标 **硬件 / 版本 / 日期 / 口径**；**完整历史测量（所有旧表、A/B、优化过程、撤回记录）
> 见 [`docs/BENCHMARKS_HISTORY.md`](docs/BENCHMARKS_HISTORY.md)**。标「历史」的表不要当现行结论引用。

### 现行能力：长上下文 64K（V100 32GB · 1.0.32 复测口径）

| seq（d=128 / 2 层小模型） | 改前 | 改后 |
| --- | --- | --- |
| 16384 | 11.97 GiB | **2.99 GiB** |
| 32768 | 跑不了（OOM） | **7.64 GiB** |
| 65536 | 跑不了（OOM） | **20.59 GiB**（真实训练步，loss 9.00 → 8.97）|

口径：Tesla V100 32GB；0.9.97 引入 query 分块 + 反向重算，README 标注为 1.0.32 复测口径；小模型 d=128 / 2 层，
真实训练步；数学上前向与物化路径 max_abs = 0.000e+00（逐位一致）。来源：[`docs/LONG_CONTEXT_RESULTS.md`](docs/LONG_CONTEXT_RESULTS.md)。

### 历史：四档执行策略的相对关系（V100 32GB · 数据 ≤ v1.0.2x · 2026-11-15）

| 档位 | 峰值显存 | 步时 | 相对 PyTorch |
| --- | --- | --- | --- |
| `speed` | 8,329 MiB | **0.24 s** | 速度 **快 2.1×**，显存 1.8× |
| `extreme` | **3,692 MiB** | **0.47 s** | 显存 **0.80×**、速度 **0.93×** |
| `memory` | **3,169 MiB** | 0.81 s | 显存 **0.68×**、速度 1.6× |

口径：模型 seq=2048 / 8 层 / d=512 / 8 头 / top_k=2 / 共享专家，真实训练步（前向+反向+AdamW）；
对照同构 PyTorch = **4,633 MiB / 0.507 s**。⚠️ **历史表**（CSA 分块之前，≤ v1.0.2x）：只用来说明档位之间的
取舍方向，**绝对倍率待用新协议复测**，不要引用成当前结论。

### 统一协议速度对照（V100 32GB · ORV 1.0.3 · 2026-11-19，唯一按新协议复测）

| seq | ORV | PyTorch | 差距 |
| --- | --- | --- | --- |
| 512 | 0.493 s | **0.313 s** | PyTorch 快 **1.58×** |
| 2048 | 0.753 s | **0.411 s** | PyTorch 快 **1.83×** |

口径：预热 + 3 次中位数、同词表（vocab 96000）、两侧都关梯度检查点、都开 AMP、12 层 MoE。
**此后 1.0.14–1.0.32 的多轮优化未按同一协议全面复测**，所以本 README 不宣称当前速度领先。
原始记录与「哪些旧数字已撤回」见 [`docs/BENCHMARKS_HISTORY.md`](docs/BENCHMARKS_HISTORY.md)。

其他必须一起说的结论（细节都在历史文件里）：

- **加速比归因**：旧表里 ORV 的领先来自 **host（CPU）端开销占比**（launch-bound），不是「纯 C 让 GPU 算得更快」；
  卡越快 / 序列越短差距越大，长序列与无 Tensor Core 卡上会收敛甚至反转（[`docs/BUG_ATTRIBUTION_2026-11-23.md`](docs/BUG_ATTRIBUTION_2026-11-23.md)）。
- **显存**：默认 `int8gpu` 的分块动量约 **3.06 B/参数**，fp32 AdamW 的 `exp_avg + exp_avg_sq` 是 **8 B/参数**（旧 4090 表口径）。
- **长序列**：seq 越长速度差距越小；注意这是历史现象，根因当时未完全归因，待复测。

---

## 能力边界与不适用场景

- **P100（Pascal / sm_60 / 无 Tensor Core）：不推荐用 ORV 训练。** 公平 2×2（2026-11-21，ORV 1.0.18）：
  ORV **慢 22–25%**；低显存注意力下显存省 12–13%，物化注意力下反而多占 9%。旧报告「显存多 26–37%」已作废
  （口径不对等）。详见 [`docs/P100_MEMORY_ATTRIBUTION.md`](docs/P100_MEMORY_ATTRIBUTION.md)。
- **没有 Tensor Core 的卡（cc < 7.0）加速路径失效**：ORV 的速度收益依赖 fp16/bf16 GEMM 与融合内核，
  P100 上这些路径全部不成立。
- **单卡容量**：1.7B 级「能跑」只在 **32 GB** 上成立（24 GB 4090 上 1.72B / seq512 会 OOM）；
  1.82B 真实 V3 形状需 5090 32GB + `int8gpu`；**3B 在 32 GB 单卡装不下**。
- **没有 ZeRO / 优化器分片**：DP 是「每卡一份完整模型 + 梯度平均」，不能把多卡显存拼成一块；
  3B 以上需要 ZeRO-2（每卡约 22.6 GiB）或 80 GB 单卡。
- **短序列开 DP 是净亏**：消费卡无 P2P，归约量 ≈ 单卡整步；实测 5090 seq512 吞吐比 0.87×、4090 0.68×，
  临界点约 seq 620。
- **AMP 不要无脑开**：1.0.14 定位并修掉负优化后，seq512 仍 +8%、seq2048 才变成 −12%（历史 4090 口径，待复测）。
- **张量语义**：`Tensor` 语义与 PyTorch 接近，但**只支持 fp32**；不支持的能力会显式报错，不会静默算错。
- **GPU 后端分发**：wheel 只带 **CPU 引擎**；GPU 版由 `tgai_go.gpu` 在目标机检测 + 编译 + 安装，
  编不出来自动回退 CPU。
- **未实测硬件**：A100 / H100 / AMD ROCm 没有本站同口径实测，只能写「未验证」；支持矩阵见
  [`docs/MEASURED_HARDWARE.md`](docs/MEASURED_HARDWARE.md)。
- **仓库根目录的历史脚本**（`tgai*.py` 等）不在 wheel 里、不承诺兼容当前 API；正式用法看 `examples/` 与 `docs/`。
- **旧数字状态**：V100 档位倍率、3090 全表、4090/5090 跨代对比、DP 全表都是历史测量，部分已撤回、
  全部待用新协议复测 —— 引用前先看 [`docs/BENCHMARKS_HISTORY.md`](docs/BENCHMARKS_HISTORY.md) 的撤回记录。

---

## 训练完怎么用：同一个引擎直接推（推训一体）

ORV 不是"只负责训练"的框架 —— **训练产物 `.TG` 就是推理输入**，中间不需要转 GGUF / ONNX。
`tgai-orv` 这个 pip 包里同时包含训练侧（`tgai_go.train`）与推理侧（`tgai_go`，即 TGAI GO）。训练完立刻能跑：

```bash
python -m tgai_go info -m model.TG        # 这个模型是什么（不加载权重）
python -m tgai_go meta -m model.TG        # 它声明的对话配置（同样不加载权重）
python -m tgai_go chat -m model.TG        # 开始对话
python -m tgai_go gen  -m model.TG "你好"  # 单次生成
```

```python
import tgai_go

e = tgai_go.TGEngine()
m = tgai_go.load("model.TG")                 # 训练产物直接加载
c = tgai_go.Chat(e, system="你是 TGAI")
print(c.ask("你好"))
```

**`.TG` 是什么**：ORV 自定义的模型容器，magic `0x49414754`，128 字节固定头，权重按**权名 FNV-1a 64** 查表
（不是按顺序），dtype 码 `6` 表示 **bf16**（当 fp16 读会静默出 NaN）。另有 `.TG4` 自描述分区容器，`read_tg` 按 magic 自动分派。

**后端**：wheel 里只带 **CPU 引擎**（GPU 二进制有平台/驱动/架构依赖）；GPU 版由 `tgai_go.gpu` 在目标机上检测 + 编译 + 安装，编不出来自动回退 CPU。

- 完整上手指南：[`docs/INFERENCE_QUICKSTART.md`](docs/INFERENCE_QUICKSTART.md)
- 终端自查：`T.help('load')` / `T.help(open='web')`；Wiki 里搜 `api:go:*`、`cli:go:*`、`fmt:TG`

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
- `nn.Module` 子类不会自动翻译：`from_torch` 只覆盖 13 种常见层，自定义 `forward` 需用 ORV 层重写（通常 10–40 行）

迁移检查清单与能力边界：[`docs/C_END_AND_MIGRATION.md`](docs/C_END_AND_MIGRATION.md)；
自定义模型最小示例：[`docs/QUICKSTART_CUSTOM_MODEL.md`](docs/QUICKSTART_CUSTOM_MODEL.md)。

---

## 分布式训练（数据并行）

模型单卡装不下、或需要更高聚合吞吐时用它：**每卡一份完整模型 + 梯度平均**，进程间走共享内存，
**不依赖 NCCL / MPI**，Linux / Windows 都能直接用。选卡用 `CUDA_VISIBLE_DEVICES`（ORV 没有选卡参数）。

```bash
CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=run1 python -m tgai_go.train.v3_pretrain --data <语料> ...
CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=run1 python -m tgai_go.train.v3_pretrain --data <语料> ...
```

- `TGAI_DP_WORLD` 总进程数（≥2 才启用）／`TGAI_DP_RANK` 本进程编号／`TGAI_DP_JOB` 必须一致。
- 归约**默认取平均**（与 PyTorch DDP 语义一致）；等效 batch 变大时按 world 放大 lr。
- **短序列加卡是净亏**（消费卡无 P2P，归约 ≈ 单卡整步），临界点约 seq 620。
- `T.DUO()`（1.0.16+）可一行接管整机：`T.DUO()` 探测 → 选方案 → 起 worker；`T.DUO(dry_run=True)` 只看计划。

完整实测与取舍见 [`docs/BENCHMARKS_HISTORY.md`](docs/BENCHMARKS_HISTORY.md) §5；运行手册：
[`docs/DP_2GPU_RUNBOOK.md`](docs/DP_2GPU_RUNBOOK.md)、[`docs/DP_OVERLAP_2026-11-21.md`](docs/DP_OVERLAP_2026-11-21.md)。

---

## 插件系统

算子、后端、损失函数、优化器、分词器、调度器 —— **六条扩展轴共用一个命名空间**，一个文件就能同时适配
CPU / CUDA / Vulkan；能力不足时**必须报错而不是静默降级**。

```python
from tgai_go.train import ops_plugin

@ops_plugin.register("my_op", backends=("cpu", "cuda", "vulkan"))
def my_op(x):
    ...
```

`.tgop` 插件包就是一个 zip（改后缀），根目录一份 `manifest.json`，可放任意多个插件；用户一行装完：
`T.plugin.install("svs-ops-0.2.0.tgop")`，再用 `T.plugin.list()` / `info()` / `uninstall()` 管理。
反向三档：组合式（自动微分）/ `register_auto`（前向原生、反向用等价实现重算）/ `register_numpy`（手写前向 + 解析反向）。

规范与示例：[`docs/PLUGIN_SPEC.md`](docs/PLUGIN_SPEC.md) · [`PLUGINS.md`](PLUGINS.md) · [`docs/PLUGIN_CHARTER.md`](docs/PLUGIN_CHARTER.md)

---

## 查文档：`T.help` 与离线 Wiki（1.0.16+）

**不要靠记忆猜 API** —— ORV 与 PyTorch 命名相近，但边界处有真实差异（例如没有 `T.sum(x)`，张量是 `x.sum()`；
没有 `Tensor.mul()`，逐元素乘用 `a * b`）。不确定就先查：

```python
T.help()                          # 总览：内核 / API / 算子 / 命令行 / 环境变量
T.help("mul")                     # 跨类别搜索（中英文正文）
T.help("mul", format="json")      # 纯 JSON（stdout 只有 JSON），给 AI / 脚本
T.help(open="web")                # 生成单文件离线 Wiki 并打印路径（可选 ?lang=zh）
```

- 清单由引擎**运行时自省**，每条带真实调用形式；拿不到的条目会明说**不可调用**，不给照抄就报错的假示例。
- 词条是外部数据 `wiki/*.txt`，改完即热更新；`T.help.check()` / `missing()` / `contrib()` / `refresh()` 维护。
- 离线 Wiki 是单个自包含 HTML：零外链、零 CDN、零外部字体，顶栏 `EN | 中文` 全局语言开关。

协议：[`wiki/README_词条规范.txt`](wiki/README_词条规范.txt)；设计：[`docs/HELP_SYSTEM.md`](docs/HELP_SYSTEM.md)

---

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [`docs/QUICKSTART_TRAIN.md`](docs/QUICKSTART_TRAIN.md) | 从安装到训练第一个模型 |
| [`docs/QUICKSTART_CUSTOM_MODEL.md`](docs/QUICKSTART_CUSTOM_MODEL.md) | 用 ORV 训练自己的模型（含能力边界） |
| [`docs/INFERENCE_QUICKSTART.md`](docs/INFERENCE_QUICKSTART.md) | 推理侧上手指南 |
| [`docs/BENCHMARKS_HISTORY.md`](docs/BENCHMARKS_HISTORY.md) | **完整历史实测与撤回记录**（README 迁出） |
| [`docs/THREE_TARGETS.md`](docs/THREE_TARGETS.md) | 三档目标的完整实测与每次优化前后对比 |
| [`docs/PYT_COMPARE_2026-11-24.md`](docs/PYT_COMPARE_2026-11-24.md) | ORV vs PyTorch：速度、显存、数学对齐、loss 曲线 |
| [`docs/PLUGIN_SPEC.md`](docs/PLUGIN_SPEC.md) | 插件包 / 插件 DSL 规范 |
| [`docs/HELP_SYSTEM.md`](docs/HELP_SYSTEM.md) | `T.help` 与 Wiki 的分层设计 |
| [`docs/ORV_KERNEL_INVENTORY.md`](docs/ORV_KERNEL_INVENTORY.md) | 引擎里已有哪些内核/入口 |
| [`docs/OP_COVERAGE.md`](docs/OP_COVERAGE.md) | 算子支持矩阵 |
| [`docs/INDEX.md`](docs/INDEX.md) | 全部文档索引（按主题分组） |
| [`CHANGELOG.md`](CHANGELOG.md) | 逐版本变更 |

---

## 项目结构

**主产物是 `python/tgai_go/train/` 这个 Python 包**（`pip install` 装的就是它）；其余目录是后端源码、构建脚本、
文档与开发工具。

```
python/tgai_go/            ← 主产物：pip install 装的就是它
  train/                   核心：自动微分 + 算子 + 优化器 + 训练循环
    __init__.py            Tensor / autograd / nn / optim / 执行档位
    cuda.py                CUDA 运行时 + 全部内核（NVRTC 源码字符串）
    kernels/               运行时现编的 .cu 算子源码（随包分发）
    v3_ops.py / v3_moe.py  V3 模型的算子与 MoE 分派
    dist.py / doctor.py    分布式 / 环境自检
  _native/                 推理引擎原生库（libtgai_engine*.dll/.so）
  wiki/                    离线 wiki 资源（随包分发）
src/ cuda/ cpu/ shaders/   原生后端源码 / CUDA 源码 / CPU 构建 / Vulkan shader
training/kernels/          训练算子源码；bench/ 微基准；examples/ 可运行示例
docs/                      设计文档与实测报告；measurements/ 原始测量日志
tools/ scripts/            维护脚本 / 一键上手脚本
```

只想用的话不需要碰 `src/` `cuda/` `shaders/`：`pip install` 装到的 Python 包已自带 NVRTC 与全部内核源码，
运行时现编译。仓库只放「能编译、能运行」所必需的东西（详见 [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md)）。

---

## 许可

MIT，见 [`LICENSE`](LICENSE)。

## 社区与示例插件

ORV 的插件系统允许第三方**不改核心**就扩展算子、层与优化器。目前社区的示例插件：

### Polaris（社区示例插件 · experimental）

自适应正交融合优化器（Muon 系：2D 权重走 Polar-Express 正交化 + 逐行 RMS + Cautious 门，1D 参数走 AdamW），
由社区用户 **yyfdfs** 开发并作为官方示例插件收录。安装 `pip install tgai-polaris`，用法
`opt = T.optim.Polaris(params, lr=1e-3)`；它的真正价值是**插件 API 的参考实现**（注册幂等、`status()` 诊断、
entry point 发现、CPU/CUDA 双路径）。

**数值现状（如实公开）**：同 lr 下末段 loss 明显不如 AdamW，且 CUDA 路径与 CPU 路径**并非完全等价**
（「逐行 RMS ≈ 整体 RMS」的近似已定位、待修）。因此它**不是**默认优化器的替代品 —— 官方默认仍是
`--opt int8gpu`。实测表与说明见 [`docs/BENCHMARKS_HISTORY.md`](docs/BENCHMARKS_HISTORY.md) §7。

- 作者主页：<https://github.com/yyfdfsyyf-source>
- 插件规范：[`PLUGINS.md`](PLUGINS.md)
