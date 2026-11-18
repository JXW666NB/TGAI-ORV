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

### 长上下文：现在能跑到 64K（0.9.97+）

CSA 的全局分支改成「query 分块 + 反向重算」后，显存从 O(seq²) 降到近似线性：

| seq（d=128 / 2 层小模型） | 改前 | 改后 |
| --- | --- | --- |
| 16384 | 11.97 GiB | **2.99 GiB** |
| 32768 | 跑不了（OOM） | **7.64 GiB** |
| 65536 | 跑不了（OOM） | **20.59 GiB**（真实训练步，loss 9.00 → 8.97）|

默认 `cq=2048`：**seq ≤ 2048 自动走原路径、逐位不变**；要完全回退设 `TG_CSA_ATTN_CQ=0`。
数学：前向与物化路径 **max_abs = 0.000e+00（逐位一致）**；4 层/seq512 的 loss 与基线精确相同。

## 分布式训练（数据并行）

模型**单卡装不下**、或想要更高聚合吞吐时用它。
ORV 的 DP 是「每卡一份完整模型 + 梯度平均」，进程间走**共享内存 + 文件屏障**，
**不依赖 NCCL/MPI** —— 所以 Linux 和 Windows 都能直接用，装完即跑。

```bash
# 两张卡、两个进程；CUDA_VISIBLE_DEVICES 指定各自的卡（ORV 没有选卡参数）
CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=run1 \
  python -m tgai_go.train.v3_pretrain --data <语料> ...
CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=run1 \
  python -m tgai_go.train.v3_pretrain --data <语料> ...
```

* `TGAI_DP_WORLD`：总进程数（≥2 才启用）
* `TGAI_DP_RANK`：本进程编号（0..world-1）
* `TGAI_DP_JOB`：**必须一致**，同一个 rendezvous 目录名

**实测（RTX 3090 24GB ×2 / V3-1.5B / seq512 / m8v16）**

| 配置 | 步时 | 聚合吞吐 |
| --- | --- | --- |
| 单卡 | 0.505 s | 1,013 tok/s |
| **双卡 DP** | 0.507 / 0.497 s | **≈ 2,039 tok/s（1.99×）** |

**双卡对双卡：与 PyTorch DDP 同条件对比**（同形状、同 seq、各 rank 不同数据、真实训练步）

| 层数 | PyTorch DDP | ORV DP | ORV 优势 | 显存（PyT / ORV）|
| --- | --- | --- | --- | --- |
| 12 | 0.905 s → 1,132 tok/s（聚合）| **0.264 s → 3,881 tok/s** | **3.43×** | 17.7 / **11.2 GiB** |
| 16 | 1.201 s → 853 tok/s（聚合）| **0.344 s → 2,976 tok/s** | **3.49×** | 22.8 / **14.2 GiB** |

单卡时 ORV 只快 2.3–2.5×，**一上双卡差距反而拉大到 3.4–3.5×** —— 原因是两边的扩展开销不同：

| | 单卡 → 双卡 每步耗时变化 |
| --- | --- |
| ORV DP | 0.252 → 0.264 s（**+5%**）|
| PyTorch DDP | 0.577 → 0.905 s（**+57%**）|

PyTorch 这边贵在两处：MoE 的路由是数据相关的，DDP 必须开 `find_unused_parameters=True`
（每步多一次图遍历），以及 NCCL all-reduce。ORV 不需要前者，后者的实测开销接近零
（`pytorch-ddp 对照` 数据见 `lab/rent/`）。
归约开销几乎为零（0.507 vs 0.505 s）。梯度归约**默认取平均**，与 PyTorch DDP 语义一致；
要让等效 batch 变大，按 world 放大 lr。正确性判据：**各 rank 的 gnorm 必须逐位相同**
（实测 3.645 / 3.526 / 5.076 两次运行完全一致）。

### V3-1.5B 规模下与 PyTorch 的对比（RTX 3090 24GB，单卡）

同形状（d=1024 / d_ff=3328 / E=4 / vocab 96000 / seq512），真实训练步：

| 层数 | PyTorch 步时 | PyTorch 峰值 | ORV 步时 | ORV 峰值 |
| --- | --- | --- | --- | --- |
| 12 | 0.577 s | 16.5 GiB | **0.252 s** | **10.5 GiB** |
| 16 | 0.818 s | 20.9 GiB | **0.328 s** | **13.2 GiB** |

**每层边际成本：PyTorch 60.3 ms vs ORV 19.0 ms ⇒ 便宜 3.2×**，显存也低约 37%。
这个形状 PyTorch 到 24 层会 OOM（fp32 AdamW 光状态就要 24 GiB 以上），ORV 用 m8v16 能跑完。

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

### 插件包：一个文件装下整个环境

`.tgop` 就是一个 zip（改后缀），根目录一份 `manifest.json`，里面可以放 **任意多个插件**。
用户只要**一行**：

```python
import tgai_go.train as T

T.plugin.install("svs-ops-0.2.0.tgop")   # ← 一行，装完整的包
T.svs.stft_mag(wav, 1024, hop_length=256)
```

开发期也可以直接指向目录：`T.plugin.install("examples/bundles/svs-ops")`。

### 看信息、卸载

```python
T.plugin.list()          # 装了什么：名字 / 版本 / 后端 / 算子数 / 命名空间
T.plugin.info("svs-ops") # 单个包的详细信息（元信息 + 它装了哪些算子）
T.plugin.uninstall("svs-ops")   # 卸载：注销算子 + 移除 T.svs
```

元信息是**强制**的：`name`、语义化 `version`、`api_version`、`backends` 缺一不可，
非法值在注册时就报错并给出修法 —— 不是写在文档里的口头约定。

### 插件的反向：三档，总能挑一档

```python
# ① 组合式：用已有算子拼，梯度由 autograd 自动给出（不用写反向）
ops_plugin.register("my_softplus", lambda x: (x.exp() + 1.0).log())

# ② 半自动：前向走原生快路径，反向自动导出 —— 只需要一份"数学等价的可微实现"
ops_plugin.register_auto("my_kernel", forward=native_call, fallback=composed_version)

# ③ 全手写：追求极致性能时给前向 + 解析反向
ops_plugin.register_numpy("my_gelu", gelu_fwd, gelu_bwd)
```

② 是**不想写反向**的人用的：训练时前向仍是原生的，反向把 `fallback` 重算一遍再反传
（所以梯度精确，不是差分近似）；推理时完全不建图。默认还带一致性检查 ——
两份实现数学不一致会直接报错，而不是悄悄算错。

---

### 一条命令装完整个环境

```python
T.plugin.install_all("bundles/")     # 目录下所有插件包一次装完
```

原生算子也**不用写反向**：manifest 里给 `form: "native"` 加一个 `impl.fallback`
（一份用已有算子写的等价实现），ORV 就自动把反向导出 —— 前向仍是你编译好的原生 kernel。
详见 [`docs/PLUGIN_SPEC.md`](docs/PLUGIN_SPEC.md)。

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

## 项目结构

**主产物是 `python/tgai_go/train/` 这个 Python 包**（`pip install` 装的就是它）；
其余目录是后端源码、构建脚本、文档与开发工具。

```
python/tgai_go/train/     ← 核心：自动微分 + 算子 + 优化器 + 训练循环
  __init__.py              Tensor / autograd / nn / optim / 执行档位（约 24 万字符）
  cuda.py                  CUDA 运行时 + 全部内核（NVRTC 源码字符串，约 35 万字符）
  v3_ops.py / v3_moe.py    V3 模型的算子与 MoE 分派
  v3_flash_op.py           分块注意力（flash 风格）
  nn_ops.py / checkpoints.py / plugin.py / ops_plugin.py
  dist.py / doctor.py      分布式 / 环境自检

src/                       原生后端 C 源码（launcher、matmul、tgai_backend）
cuda/                      .cu 后端（与 NVRTC 运行时并存的另一条路径）
cpu/                       CPU 后端构建
shaders/                   Vulkan compute shader（.comp 与编译产物 .spv）
embedded/ tcc/ third_party/ 嵌入式运行时、TCC 工具链、Vulkan 依赖

tests/                     测试（含 fixtures 与算子对拍）
bench/                     逐模块微基准（一个模块一个文件，CUDA event 计时）
tools/                     仓库维护脚本（发版、探针、远程运维）
examples/                  可直接运行的示例
scripts/                   一键上手脚本
docs/                      设计文档、计划、实测报告
.github/                   CI（单元测试 + 打包自检）
```

各平台的构建脚本在根目录：`build_cuda*.bat`、`build_vulkan*.bat`、`rebuild_cpu.bat`；
发版用 `build_dist.bat` 与 `publish_pypi.bat`。

> 只想用的话不需要碰 `src/` `cuda/` `shaders/`：`pip install` 装到的 Python 包已自带
> NVRTC 与全部内核源码，运行时现编译。

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
