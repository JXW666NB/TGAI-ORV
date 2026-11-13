---
name: orv-engine-dev
description: 开发、迁移与扩展 TGAI ORV 训练引擎（PyPI 包 tgai-orv，import 路径 tgai_go.train）。当用户要为 ORV 写模型/算子/插件、把 PyTorch 训练代码迁移到 ORV、排查 ORV 的训练或显存问题、或需要确认 ORV 到底支持什么时使用。也适用于为 ORV 补后端（CUDA/Vulkan/ROCm）与写插件 DSL 生成器。
---

# ORV 引擎开发

本 skill 面向**替代你原本打算写 PyTorch 代码的场景**。核心原则：
**先查证 ORV 有什么，再写代码** —— ORV 与 PyTorch 有真实差异，凭 PyTorch 记忆写必然出错。

---

## 1. ORV 是什么

纯 Python 的自动微分 + 训练引擎，**不依赖 PyTorch**，直接走 CUDA Driver API + cuBLAS + NVRTC。
PyPI 包名 `tgai-orv`，import 路径 `tgai_go.train`。

它从 TGAI GO（推理引擎）长出来；GO 可理解为只保留推理的精简版 ORV。

**已实测的差异化**（不要夸大成别的数字）：3B MoE（2.97B）全参一步 709 ms / 峰值 24.78 GiB
（V100 32GB）；同条件下 PyTorch 在 1.997B 就 OOM。

---

## 2. 铁律（违反会直接出错，务必先读）

1. **不要凭 PyTorch 记忆写 ORV 代码。** 写之前先看 `docs/CAPABILITY_AUDIT.md`（脚本生成的清单）
   与 `docs/OP_COVERAGE.md`。ORV 名字与 PyTorch 对齐的部分很多，但**边界处有 8 项真实差异**（见 §6）。
2. **不把"跑通了"当"验证了"。** 交付前必须给出可复现命令与数字。
3. **不把"未验证"写成"已支持"。** 多卡、A100/H100 bf16、7B+ 模型、Vulkan 训练，全部【未验证】。
4. **热路径不许有 Python 逐元素。** 逐元素循环必须下沉到原生内核或组合算子。
5. **插件可以更快，不可以"更准"或"更省事地不同"。** dtype 提升、累加顺序、边界行为（nan/inf/空张量）
   属于核心语义，插件不得改变。
6. **改代码前先看 `docs/INDEX.md`。** 仓库有 70+ 篇文档，不先读会写出与既有结论冲突的东西。

### 2.1 硬件已知限制（外部实测，2026-11-12 · P100）

现场已验证的架构（截至 2026-11-12）：

| 架构 | 卡 | 结论 |
| --- | --- | --- |
| **sm_60** | Tesla P100 16GB | NVRTC 能编 `compute_60`，正常训练 —— 硬件不是障碍 |
| **sm_61** | GTX 1060 6GB | 外部用户用它训出了 aisa1-3；30M 级模型 <1 GB 显存 |
| **sm_70** | Tesla V100 32GB | 3B MoE 全参一步 24.78 GiB / 709 ms（AdamW） |

一位用户在 **P100 16GB（sm_60，无 Tensor Core）** 上实测，结论要先知道，别当成普适能力：

| 项 | 事实 |
| --- | --- |
| **sm_60 本身支持** | NVRTC 能编译 `compute_60`，张量/模型/AdamW 正常训练 —— 硬件不是障碍 |
| **sm_61（1060 等消费卡）上融合注意力反向不可用** | `_ATTN_FLASH_SMEM = 68 KB`、`_ATTN_FLASH_G2_SMEM = 84 KB`，而 **sm_61 单 block 动态 smem 上限只有 48 KB 且不支持 opt-in** ⇒ `lowmem=True` 前向正常、**反向抛 `cuFuncSetAttribute(attn_bwd_flash maxdyn) rc=1 invalid argument`**。默认 `lowmem=False` 时不受影响（外部实测，2026-11-12）|
| **Windows 必须自备 CUDA Toolkit 12.x（或 torch 的 lib）** | 驱动目录里**没有** NVRTC / cuBLAS；`tgai-orv[cuda]` 这个 extra **只对 Linux 生效**。所以"有驱动就不用装 Toolkit"在 Windows **不成立**（外部实测）|
| **首装可能报 `failed to open nvrtc-builtins64_*.dll`** | **0.9.50 已修**（会扫 `nvidia/*/lib`、`*/bin` 并写进 PATH）。仍失败时的规避：把 `site-packages/torch/lib` 前置进 `PATH` **并** 调 `os.add_dll_directory` |
| **显存：开不开检查点差一个量级** | 实测（D=1280/L=12/dh=128/FF=5120/S=1024/B=4，fp32+AdamW）：**不开检查点** ORV 19.58 vs torch 8.16 GiB（**2.40×**）；**开检查点** ORV 8.02 vs torch 7.56（**1.06×**）。权重/梯度/优化器三类两边**字节完全相同**，差距 100% 在激活。⇒ **检查点不是可选项**，`docs/MEMORY_ATTRIBUTION_2026-11-12.md` |
| **检查点 0.9.53 起与 PyTorch 语义对齐** | 旧版把 `_parents` 清空，导致外层图段间断开、反向递归整链重算。按 `torch/utils/checkpoint.py` 的 `CheckpointFunction` 重做后：每层反向峰值 **1.68 → 0.117 GiB**；20 层/606M/B=2/S=2048 由 **31.29 GiB 降到 7.56 GiB**（`docs/MEMORY_BACKWARD_PEAK_2026-11-12.md`）|
| **"能装下更大模型"的准确含义** | 指 PyTorch 在 **1.997B 稠密**就 OOM（31.40 GiB）而 ORV 能跑 **3B MoE**（24.78 GiB）。这是融合减少中间量，不是"更省显存" |
| **fp16 是「反向也走」但受形状门槛** | fp16 分支在**共享 GEMM 核心**里，前向反向都走；门槛 `m*n ≥ 262144` 且 `k ≤ _FP16_MAX_K`（默认不限，可用 `TG_FP16_MAX_K` 调）。**长 K 的 dW 有精度风险**：代码记录梯度相对误差会从 3e-4 恶化到 2e-2。`B=8/S=2048` 时 k=16384 正处风险区 ⇒ 开 fp16 必须在目标卡实测 |
| **`.TG` 的 v4 现在是 TG4 容器** | 0.9.60 起：**检查点与导出写 TG4**（magic `0x00344754` = `TG4\0`，128B 头 + section table + `NAM` 名字表 + 内置分词表）。读取端 `read_tg()` **按 magic 分派**：v3 家族（`0x49414754`，含"v3 布局 + version=4"的旧文件）与 TG4 都能读；引擎侧同样两条分支（`TG_IS_V3FAMILY` / `tg4_attach`）。**注意**：`magic=0x49414754` + `version=4` 是**旧**写法，新文件不会是它 |
| **`.TG` 有**两套**"v4"，别混** | ① `pack.py::TGWriter(version=4)` = **v3 布局 + 头里写 4**（magic `0x49414754`）—— **旧**检查点与 V2.7 的 `*_v4.tg` 是它，引擎仍能读；② `tg4.py` = **TG4 容器**（magic `0x00344754` + section table + `NAM`），**0.9.60 起是检查点/导出的默认**。排查时先看 magic：`0x49414754` 走 v3 家族，`0x00344754` 是 TG4 |
| **v3 ↔ TG4 转换** | v3 → TG4 用 `tools/to_tg4.py`（需名字边车；流式，峰值=单张量）；只改版本号用 `tools/set_tg_version.py`（只动 4 字节，遇 TG4 会拒绝）。TG4 落地前已扫盘确认真实文件为 0 个，故无历史迁移负担 |
| **AMP 不省激活显存** | 参数/激活/梯度**全程 fp32**，只有 GEMM 输入 cast。torch autocast 把激活存 fp16 ⇒ 天然省一半激活显存。**这是 ORV 显存差距的结构性原因之一**，不是"开销没优化好" |
| **`sdpa_lowmem` 只支持 `dh==64`** | `dh=128` 会**静默退回全量注意力**（不报错）。d_head=128 的模型别以为开了低显存 |
| **没有 `cos` / `sin`** | **0.9.51 已加**。旧版本需 numpy 预生成表 |
| **SDPA 维度约定** | 已接受 `is_causal=` 与 `[B,H,S,Dh]`（0.9.51 起自动展平）；旧版本要 `[B*H, S, Dh]` |

### 2.2 多设备：现在只有单卡
**ORV 目前只保证单卡。** 下面这些**都不存在**，不要按它们设计：

| 常被误以为有的 | 真实状态 |
| --- | --- |
| 多厂商交火 / 显卡间 P2P（"TG Link"） | **只是文档里的设想，未实现、未验证**。`docs/MULTIVENDOR_ROADMAP.md` 是路线设想，不是能力说明 |
| 一个进程里混用多种后端 | **未设计、未实现**（`_CudaAPI` 是进程级单例，要支持得先拆它） |
| 同后端多张同厂商卡同进程 | 也没有。`dist.py` 是**多进程**、per-rank，且**未在真多卡上验证过** |

**不要向用户宣称** ORV 支持上述任何一种。要做多卡，目前只有 `dist.py` 的多进程路线，
且需要自己在目标机器上先验证。

#### 但要区分：「多卡代码存在」≠「多卡能力已验证」

仓库里**有**两条数据并行实现，验证程度不同 —— 向用户表述时按这个口径：

| 模块 | 机制 | 验证到哪 |
| --- | --- | --- |
| `train/dist.py` | 主机中转（梯度 D2H → 共享内存 → 汇总 → H2D），**与设备无关** | **正确性已验证**：CPU 两进程 + **同一张 V100 上两个 rank** 对拍（广播一致、梯度平均 2.0 / 缺梯度按 0 补成 2.5）。**加速比没验**（无双卡机） |
| `train/dist_p2p.py` | GPU 直连（CUDA IPC + `cuMemcpyPeerAsync`） | **完全未验证**：单卡上 `cuMemcpyPeerAsync` 没有意义，必须双卡 |
| TP / PP | — | **未实现** |

`dist.py` 在 2026-11-13 修掉 6 个 bug，其中两个**静默**（CUDA 广播无效、缺梯度侧 grad 变 CPU 数组）
和两个**卡死**（`close()` 删标记把慢 rank 等死；缺梯度时各 rank 字节数不同导致共享内存名字不同）。
细节与复现命令见 `docs/DISTRIBUTED_STATUS_2026-11-13.md` —— **写多卡相关结论前先读那篇**。

完整报告与复现步骤：`docs/REVIEW_P100_2026-11-12.md`。

---

### 2.3 大模型全参训练的显存门槛（V2.7 = 2.999B 实测）

**32 GB 卡上，全参训练的可行线是 fp16/bf16 的「权重+梯度+优化器状态」= 22.35 GiB**：

| 方案 | 权重 | 梯度 | Adam m/v | 合计 | 32 GB 能否 |
| --- | --- | --- | --- | --- | --- |
| fp32 全套 | 11.18 | 11.18 | 22.36 | **44.7 GiB** | ✗ |
| fp32 权重/梯度 + fp16 状态 | 11.18 | 11.18 | 11.18 | **33.5 GiB** | ✗ |
| fp16/bf16 权重+梯度+状态 | 5.59 | 5.59 | 11.18 | **22.35 GiB** | ✓（留激活）|

**实测过的事实**（别凭印象说）：

* **PyTorch 原生 `AdamW` 是 fp32 状态** ⇒ 2.999B 要 39.11 GiB，实测三档 seq(512/1024/2048) 全 OOM，
  死点在 `torch/optim/adamw.py::_init_group`（模型和梯度已建好，崩在分配 `exp_avg_sq`）。
* **ORV 0.9.62 的 `V27Trainer.make_optimizers()` 返回 `T.optim.AdamW`（fp32）**，参数也以 fp32 存放
  ⇒ 44.7 GiB，同样三档全 OOM。**只把优化器换成 `Fp16MomAdamW` 只到 33.5 GiB，仍然放不下** ——
  必须同时把权重/梯度降到 fp16。
* 改 fp16 参数的落点是三个类的流式装载方法（`Linear` / `MoE` / `RMSNorm` 的
  `load_state_dict_stream` 里的 `h2d` dtype）+ `V27Trainer.__init__` 的 `param_dtype` 透传；
  **别去改 `load_layer`**（它只喂 numpy 切片，精度不在那儿定）。head/final_norm/RoPE 表保持 fp32。

细节与语料格式坑（V2.6 旧标记 vs V2.7 `<用户>` 标记）见 `docs/V27_SFT_BENCH_2026-11-13.md`。

## 3. 三条最容易踩的坑（真实发生过）

| 坑 | 事实 |
| --- | --- |
| `min` / `max` 被遮蔽 | ORV 模块内 `min`/`max` 是**张量 API**。做标量比较必须用 `_builtin_min` / `_builtin_max` |
| 参数名不一致 | `Tensor.sum(axis=…)` 用 `axis`；`T.cat(tensors, dim=…)` 用 `dim` |
| `cuda.py` 不是普通模块 | 它被 `exec` 进私有命名空间：`tgai_go.train.cuda` 是 `_CudaAPI` 实例，`sys.modules` 里没有该模块 |

另外：CPU 张量的 `.grad` 是 numpy 数组；CUDA 张量的 `.data` 是 `None`（数据在 `_gpu`）——
取值统一用 `.to_numpy()`。

---

## 4. 怎么用（最小闭环）

```python
import tgai_go.train as T

text  = T.load_corpus("my_corpus.txt")                 # 纯文本或 .jsonl
tok   = T.train_tokenizer(text, vocab_size=4000)       # 无需外部 tokenizer
data  = T.tokens(tok.encode(text), seq=128)
model = T.preset.transformer(vocab=tok.vocab_size, d_model=256, n_layers=6, n_heads=8)
hist  = T.fit(model, data, steps=2000, lr=3e-4, amp=True)
```

自定义模型只需：`parameters()` + `forward(x, y) -> loss`。

完整脚本见 `examples/beginner_pipeline.py`（不依赖 GPU）。

---

## 5. API 面（先查再写）

| 类别 | 有什么 |
| --- | --- |
| 张量 / autograd | `Tensor`、广播、`backward()`、`requires_grad`、`no_grad` |
| 张量搬运 | **只有** `.cuda()`、`.to(device)`、`.cpu()`、`.to_numpy()`。**没有 `.to_cuda()`** |
| `nn` | `Module`、`Parameter`、`Linear`、`Embedding`、`LayerNorm`、`RMSNorm`、`Conv1d/2d`、`Dropout`、`BatchNorm1d/2d`、`MaxPool/AvgPool`、`Sequential`、`ModuleList`、`MoE`、`CausalSelfAttention` |
| 优化器 | `AdamW`、`SGD`、`Adafactor`、`LAMB`、`Lion`、`Muon`、`Accumulator`、`scale_grads` |
| 调度 | `CosineLR`、`LinearLR`、`StepLR`、`warmup` |
| 训练工具 | 梯度裁剪 `clip_grad_norm_(params, max_norm)`、梯度累积、`DataLoader`（`Dataset`/`IterableDataset`/`Sampler`/`collate`/多线程） |
| 精度 | `T.amp()` / `amp_on()` / `amp_off()` / `amp_state()`。**语义与 `torch.autocast` 不同**：只把 **GEMM 输入** cast 成 fp16/bf16，**参数/激活/梯度全程 fp32** ⇒ 不省激活显存、不需要 GradScaler。门槛是 `m*n ≥ 262144` 且 `k ≤ _FP16_MAX_K`（`TG_FP16_MAX_K` 可调） |
| 注意力 | `functional.scaled_dot_product_attention(q,k,v, causal=|is_causal=, scale=, lowmem=)`，输入 `[B,H,S,Dh]`。低显存内核**只支持 `Dh==64`**；`sdpa_lowmem_ready()` 无参数，只表示"本机有该内核"，**不按形状判断** |
| 索引 / 收集 | `x[idx]` 只支持**首轴一维整型**（CPU 与 CUDA 都可，0.9.52 起）；另可用 `gather_flat(x, flat_index, n_src=None)`、`index_select(x, dim, index)`。多维花式索引与布尔掩码不支持 |
| 生态互转 | `from_torch(module)`、`to_torch(module)`、`state_dict_from_torch(module)`、`state_dict_to_torch(module)`、`save(obj, f)`、`load(f, map_location=None)` |
| 规模 | 见 `docs/CAPABILITY_AUDIT.md`（脚本生成，数字会变，**不要在这里抄死**） |

**权威来源**：`docs/CAPABILITY_AUDIT.md`（脚本扫描生成，可重新生成）；`docs/OP_COVERAGE.md`（算子矩阵）。

---

## 6. 从 PyTorch 迁移：标准四步

**不要一次改完。每一步都能停下来验证。**

### 第 1 步：只换 import

```python
# import torch as T
import tgai_go.train as T
```

### 第 2 步：搬权重，不要重训

```python
model = T.from_torch(torch_model)
# 或只搬权重（注意：参数是 torch 的 module，不是 state_dict）：
mine.load_state_dict(T.state_dict_from_torch(torch_model))
```

**用已发布的 pip 包（没有仓库）时的最小验证脚本** —— 只用 `tgai_go.train` + numpy，
20 步小模型，打印 loss 与峰值显存。**这段在 V100 上逐字实跑过**
（`loss 6.238792 → 6.176723`，峰值 `0.077 GiB`，loss 下降即环境正常）：

```python
import time
import numpy as np
import tgai_go.train as T
from tgai_go.train import functional as Fn

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)
V, D, L, NH, FF, S, B = 512, 256, 2, 4, 1024, 128, 4
DH = D // NH

def dev(n, m):
    return T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.02, (n, m)).astype(np.float32)))

qkv = [dev(3 * D, D) for _ in range(L)]
o = [dev(D, D) for _ in range(L)]
g = [dev(FF, D) for _ in range(L)]
u = [dev(FF, D) for _ in range(L)]
dn = [dev(D, FF) for _ in range(L)]
head = dev(V, D)
emb = T.nn.Embedding(V, D)
params = [emb.weight, head] + qkv + o + g + u + dn
for t in params:
    t.requires_grad = True

ids = rng.integers(0, V, (B, S)).astype(np.int64)
tgt = rng.integers(0, V, (B, S)).astype(np.int64)
x = emb(ids.reshape(-1)).reshape(B, S, D)
for i in range(L):
    z = x.reshape(-1, D) @ qkv[i].T
    q, k, v = z.reshape(B, S, 3, NH, DH).permute(2, 0, 3, 1, 4)
    y = Fn.scaled_dot_product_attention(q.reshape(B * NH, S, DH),
                                        k.reshape(B * NH, S, DH),
                                        v.reshape(B * NH, S, DH), is_causal=True)
    x = x + (y.reshape(B, NH, S, DH).permute(0, 2, 1, 3)
             .reshape(-1, D) @ o[i].T).reshape(B, S, D)
    f = x.reshape(-1, D)
    x = x + ((Fn.silu(f @ g[i].T) * (f @ u[i].T)) @ dn[i].T).reshape(B, S, D)

opt = T.optim.AdamW(params, lr=1e-3)
for step in range(20):
    loss = Fn.cross_entropy((x.reshape(-1, D) @ head.T).reshape(-1, V), tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    rt.sync()
    if step in (0, 19):
        st = rt.pool_stats()
        print("step %2d loss %.6f 峰值 %.3f GiB"
              % (step, float(np.asarray(loss.to_numpy()).reshape(-1)[0]),
                 st["peak_live_bytes"] / (1 << 30)))
```

**搬完必须对拍前向**：

```python
y_t = torch_model(x).detach().numpy()
y_o = mine(T.tensor(x)).to_numpy()
rel = abs(y_o - y_t).max() / max(abs(y_t).max(), 1e-9)     # 期望 ≤1e-5
```

### 第 3 步：改自定义 `forward`

**这是唯一无法自动迁移的部分。** `from_torch` 遇到自定义 `nn.Module` 子类会报错并列出支持的类型
—— 这是刻意设计：猜 `forward` 语义会静默算错。用 ORV 的层重写，通常 10~40 行，**改完再对拍一次**。

### 第 4 步：接训练循环

```python
opt  = T.optim.AdamW(model.parameters(), lr=3e-4)
hist = T.fit(model, loader, steps=2000, lr=3e-4, amp=True)
```

### 迁移检查清单

- [ ] 只换 import 就能跑通前向
- [ ] 权重搬完，前向对拍 ≤1e-5
- [ ] 自定义 `forward` 改写完，再次对拍 ≤1e-5
- [ ] 反向也对拍（`dL/dx` 与权重梯度，容差可到 1e-4）
- [ ] **不搬 torch 的优化器 state**（动量含义可能不同）
- [ ] 单步耗时与显存是**实测**的，不是从 PyT 外推

### 与 PyTorch 的 8 项真实差异

| # | 差异 | 处理 |
| --- | --- | --- |
| 1 | 仅单卡，无 DDP/FSDP（多卡代码有但【未验证】） | 多卡不在当前保证范围 |
| 2 | 自定义 `forward` 无法自动迁移 | 用 ORV 层重写 |
| 3 | `Tensor.max(dim=)` 只返回值，不返回索引 | 需要索引用 `argmax` |
| 4 | CUDA 上 `permute/reshape` 是**真拷贝**（PyT 是视图） | 大张量少 permute，否则显存涨 |
| 5 | `.to(device)` / `.cuda()` / `.cpu()` 是**叶子级搬运，会断 autograd 链** | 只用于数据搬运，别放 forward 中间 |
| 6 | 无 `torch.compile` | 加速靠 `T.amp()` 与逐层 CUDA Graph |
| 7 | AMP 语义不同：参数/激活/梯度全程 fp32，仅 GEMM 输入 cast | **不需要 GradScaler** |
| 8 | 无 torchvision/torchaudio | 只做训练核心 |
| 9 | **CPU 上 `opt.step()` 之后 `p.data` 变成 Tensor**，`p.to_numpy()` 会返回 Tensor 而不是 ndarray（CUDA 上正常）| 一律 `np.asarray(p.to_numpy(), np.float32)`；或统一在 CUDA 上取 |
| 10 | **pip 装的 wheel 不含推理用的原生库**（`_native/*.dll`），`tgai_go.load()` 会 `FileNotFoundError` | 推理要用源码装（`pip install -e .`）；这是已知打包缺陷（外部实测）|

---

## 7. 开发插件

先读 **`docs/PLUGIN_CHARTER.md`** —— 那是规范性文档（不变量），与它冲突的代码视为 bug。
入门看 `docs/PLUGINS.md`。

### 六条轴

| 轴 | 注册 | 生效 |
| --- | --- | --- |
| 算 子 | `plugin.op(name, fn)` / `op_numpy` / `op_c` / `op_native` | 立即 |
| 后 端 | `plugin.backend(name, factory, caps=…)` | 需 `activate()` |
| 优化器 | `plugin.optimizer(name, cls, works_inplace=…)` | 立即 |
| 层 | `plugin.layer(name, cls)` | 立即 |
| 骨架 | `plugin.arch(name, builder)` → `T.build(name, …)` | 立即 |
| API 兼容层 | `plugin.api(name, installer)` | 需 `activate()` |

支持装饰器写法（`@plugin.op("x")`），一个 `.py` 文件可装完全部六轴。

### 四档可移植性（宪法 §3）

| 档 | 形态 | 可移植 |
| --- | --- | --- |
| **B1 声明式内核** | 用户写数学，ORV 生成各后端代码并编译 | ✅ 全后端（**S1 未实现**）|
| B2 后端源码 | CUDA C / GLSL / 厂商语言 | ❌ 该后端 |
| B3 预编译库 | `.so`/`.dll` + C ABI 或 JSON 清单 | ❌ 该库架构 |
| P 组合式 | 用已有算子拼（Python） | ✅ 全后端 |

**纪律**：能由 B1 表达的自定义算子**不得**只提供 P 档；缺生成器的后端必须让插件
**表现为不存在**，不得静默降级；P 档兜底必须可统计。

### 性能纪律

分派开销实测 **627 ns**。大 GEMM 占 0.16%（可接受），小 elemwise 占 4.2%（**超标**）。
热路径必须静态绑定：

```python
f = plugin.bind("my_op", backend, caps=caps, arch=cc, dtype="fp32")
for _ in range(steps):
    y = f(x)
```

### 开发插件时最容易犯的错

- 自定义层/优化器/骨架的名字必须是**合法标识符**（会挂成属性）；`backend`/`api` 允许连字符
- 重名登记会报错，覆盖要显式 `replace=True`
- 出式更新的优化器（`p.data = p.data - …`）**不能**用于 `ParameterGroup`（会断开视图共享）
- 加载原生库 = **执行任意机器码**，无沙箱，必须在文档里写明

---

## 8. 训练完之后：用内置 TGAI GO 推理

训完的模型可以**直接推理**，不需要转成别的框架格式。先明确一件事：

> **两种安装形态能用到的命令不一样。**
> `pip install tgai-orv` 装出来的包里**只有 `python -m tgai_go`**（wheel 里没有 console entry point，
> 也没有 `tgai_tui.py`）；`tgai.bat` / `tgai_tui.py` / `tgai_chat.py` / `tgai_ui.py` 都在**源码仓库根目录**，
> 只有 clone 下来才有。

### 第 1 步：导出成 `.TG`

```python
from tgai_go.train.export import save_tg

save_tg("my_model.TG", params,          # params: {训练侧名字: numpy 数组}
        n_layers=28, d_model=1024, n_heads=16, d_ff=4096,
        vocab_tokens=tok.vocab, merges=tok.merges,
        max_seq=512, n_experts=8, n_activated=2, rope_theta=10000.0)
```

`params` 用**训练侧命名**（`experts.{e}.gate/up/down`），导出时映射成引擎的 `w1/w2/w3`。
**0.9.60 起导出写的是 TG4 容器**（built-in 名字表 + 分词表），引擎从 0.9.58 起就能读。

### 第 2 步：三种跑法（按你手上的东西选）

#### A. 只有 pip 包 —— 用 `python -m tgai_go`

```bash
python -m tgai_go version                       # 版本 + 原生库是否就位
python -m tgai_go info  -m my_model.TG          # 层数/宽度/词表/上下文（会加载权重）
python -m tgai_go meta  -m my_model.TG          # 只看内嵌对话配置（不加载权重）
python -m tgai_go gen   -m my_model.TG "你好" -n 128 -t 0.6 --top-k 40 --top-p 0.9
python -m tgai_go chat  -m my_model.TG -s "你是一个简洁的助手"
```

* `gen` = **单次生成**，打印完整回复；
* `chat` = **交互 REPL**（流式输出），内置 `/exit`（退出）、`/clear`（清历史与 KV cache）、`/stats`（轮数/线程/上下文上限）；
* 通用参数：`-n/--max-tokens`、`-t/--temperature`、`--top-k`、`--top-p`、`--threads`、`--stop`（可多次）、`--preload`。

#### B. clone 了仓库 —— 有 TUI 和服务

```bash
python tgai_tui.py -m my_model.TG --sys "你是一个简洁的助手"   # 全屏 TUI（依赖同目录 tgai_chat.py / tgai_ui.py）
tgai.bat chat -m my_model.TG        # 同上，Windows 一键（不带参数会自动挑模型）
tgai.bat server -m my_model.TG      # 只开 OpenAI 兼容服务
tgai.bat bench                       # 基准测试
tgai.bat convert ...                 # .pt / .gguf → .TG
tgai.bat rebuild                     # 重编引擎（等价于仓库根的 build_engine.bat）
tgai.bat selfcheck                   # 自检
tgai.bat help                        # 完整帮助
```

TUI 内斜杠命令（已核实存在）：`/help`（全部命令）、`/params`（调参）、`/exit`。

#### C. 写在 Python 里

```python
import tgai_go

with tgai_go.load("my_model.TG") as m:            # dll= 可显式指定引擎库
    print(m.n_layers, m.d_model, m.vocab_size, m.max_seq_len)

    # 单次
    print(m.generate("你好", max_tokens=64, temperature=0.6, top_k=40, top_p=0.9))

    # 多轮（带历史与 KV cache），流式
    s = m.session(system="你是一个简洁的助手")
    for chunk in s.stream("讲个故事", max_tokens=64):
        print(chunk, end="", flush=True)
    s.clear()

    ids = m.encode("你好"); print(ids, m.decode(ids))
```

### 训练检查点可以**直接**当模型跑

`ckpt.save_checkpoint` 写出的 `weights.tg`（TG4，含分词表）就是一个可加载的模型 ——
不必先导出。实测（V100）：`tgai_go.load(".../step_000002/weights.tg")` 得到
`layers=28 d_model=1024 vocab=32779`，`encode('你好')` 与 `generate` 都正常。

```python
with tgai_go.load(r"D:/runs/step_000002/weights.tg") as m:
    print(m.generate("你好", max_tokens=32))
```

### 导出时三个会让模型"能加载但生成崩"的坑

| 坑 | 症状 | 处理 |
| --- | --- | --- |
| **RoPE 的 cos/sin 没填** | 加载正常，`generate` 时 `access violation reading 0x…` —— 引擎**按指针直接读**这块 buffer | `collect()` 已按公式补齐，别绕过它自己拼 cfg |
| **`head_dim != d_model // n_heads`** | v3 容器头里**没有 head_dim 字段**，引擎按 `d_model//n_heads` 解析 ⇒ 导出必错且不报错 | **TG4 头里有独立 `head_dim`**（引擎会用它），但 `save_tg` 目前仍对非自然 head_dim 报错 —— 要放开需先在目标模型上实测 |
| **RoPE theta 与训练侧不一致** | 位置编码错位，输出乱 | 训练用什么 theta，导出就传什么 |

### 交付前必须验证的三件事

1. **加载不崩**：`python -m tgai_go info -m my_model.TG` 能打印层数/宽度
2. **生成不崩**：`python -m tgai_go gen -m my_model.TG "你好"` 能出完整回复（RoPE buffer 缺失在这一步才暴露）
3. **数值对得上**：同一 prompt 下，训练侧前向的 logits 与引擎输出的 top-k 一致或高度接近

漏掉第 2 步是最常见的翻车方式 —— 加载成功会让人误以为导出正确。

---

## 9. 格式互转与「拿别的模型来训练」

支持的格式与方向（接口均已在代码中确认）：

| 格式 | 读（当训练起点） | 写（导出） |
| --- | --- | --- |
| **`.TG`**（TGAI 自有） | ✅ `checkpoints.load_tg` | ✅ `export.save_tg` |
| **GGUF** | ✅ `checkpoints.load_gguf`（仅 F32/F16/BF16） | 见 `tgai-to-gguf` skill |
| **`.safetensors`**（HF 目录） | ✅ `checkpoints.load_safetensors_dir` | — |
| **`.pt`**（torch） | ✅ `interop.from_torch` / `state_dict_from_torch` | ✅ `interop.to_torch` |
| **`.npz`** | ✅ `checkpoints.load_npz` | ✅ `checkpoints.save_npz` |
| **ORV 自己的检查点** | ✅ `ckpt.load_checkpoint` | ✅ `ckpt.save_checkpoint` |

### 9.1 `.pt` / `.gguf` → `.TG`

```bash
tgai.bat convert <args>          # 实际调用 tools/tgai_convert.py
```

命令行细节见 `tgai.bat convert --help`、`docs/GGUF_TO_TG.md`。

Python 侧也可以自己走：

```python
from tgai_go.train.export import save_tg
save_tg("out.TG", params, n_layers=…, d_model=…, n_heads=…, d_ff=…,
        vocab_tokens=vocab, merges=merges, max_seq=512,
        n_experts=8, n_activated=2, rope_theta=10000.0)
```

### 9.2 `.TG` → GGUF

走 `tgai-to-gguf` 这套流程（llama.cpp 侧转换）。要点与坑（乱码/循环）见该 skill 与
`docs/GGUF_TO_TG.md`。

### 9.3 把 `.TG` 当训练检查点（继续训练 / 微调）

```python
from tgai_go.train import checkpoints

ck = checkpoints.load_tg("base.TG")            # → Checkpoint（权重 + 结构配置）
# 只想拿几层对拍、省内存时：
ck = checkpoints.load_tg("base.TG", max_layers=4, vocab_limit=32768)
```

`load_tg` 会做 **norm 合理性检查**；确认无误后可传 `allow_suspicious_norms=True` 跳过。

### 9.4 把 GGUF 当训练检查点

```python
ck = checkpoints.load_gguf("model.gguf")
```

**限制（会明确报错，不是静默失败）**：

- **只有 F32 / F16 / BF16 能直接读**；量化类型（Q4_K 等）会抛出 `CheckpointError` 并给出修法
- 需要 `pip install gguf`

### 9.5 把 `.pt` 当训练检查点

```python
import tgai_go.train as T
from tgai_go.train import interop

mine = interop.from_torch(torch_model)          # 结构 + 权重一起搬
# 或只搬权重：
mine.load_state_dict(interop.state_dict_from_torch(torch_sd))
```

`from_torch` 按层类型逐个翻译（`Linear`/`Sequential`/`ModuleList`…）；
遇到它不认识的类型会**报错并列出支持的类型** —— 这是故意的，猜会静默算错。

### 9.6 ORV 自己的检查点（训练中断续跑）

```python
from tgai_go.train import ckpt
ckpt.save_checkpoint(dir, tr, st, opts, head_opt, names, src_tg, step, cfg)
ckpt.load_checkpoint(dir)
ckpt.list_checkpoints(...)        # 配合 rotate() 做滚动保留
```

它会同时保存优化器状态（`opt_fp16mv.save_opt_states`），所以续跑能接上动量。

### 9.7 迁移别人的模型来训练：三个必须核对的点

1. **名字映射**：GGUF 有 `map_gguf_name`、HF 有 `hf_name_to_ours`，都集中在 `checkpoints.py`，
   改架构只改映射表。
2. **Linear 布局**：`.TG` 里有 permute 约定，`checkpoints` 提供 `_tg_perm` / `fix_linear_layout`
   自动修正 —— 但**必须跑一次前向对拍**确认，别信"加载成功"。
3. **norm sanity**：`norm_sanity()` 会检查 norm 层权重是否离谱。跳过了就可能训出一个发散模型。

---

## 10. 排查手册

| 现象 | 先查什么 |
| --- | --- |
| 显存不够 | `T.preset` 的梯度检查点；`docs/ROADMAP_3B_PERF.md` 的实测账 |
| 数值对不上 | 先确认 dtype 与容差口径；再查 `docs/GAP_VS_PYTORCH.md` §2（注意该文档已加时效声明）|
| 速度慢 | 用 CUDA event 计时（Windows `time.time()` 粒度 15.6 ms，不可用）；确认预热 ≥3 次、输出缓冲预分配 |
| 分组 GEMM 相关 | 已归档为否定结论：`cublasSgemmGroupedBatched` rc=7，见 `docs/MOE_MODERN_ARCH.md` §3.1 |
| Muon 慢 | 已修（0.9.49，35 s→5.78 s）；但 3B 仍建议 AdamW |

**测量纪律**：所有性能数字必须用 CUDA event，且预热充分。历史上同一个量三次测得
6.11 / 6.42 / 4.18 ms，前两次都把输出缓冲分配算进了计时循环。

---

## 11. 参考文档（按用途）

| 需求 | 文档 |
| --- | --- |
| **想搞多厂商交火 / 进程内混后端** | `docs/MULTIVENDOR_ROADMAP.md` —— **只是路线设想，未实现**。别据此向用户承诺能力 |
| 想帮忙/了解现状 | ★ `docs/COMMUNITY_ONBOARDING.md` |
| ORV 到底有什么 | ★ `docs/CAPABILITY_AUDIT.md` |
| 长期设计与技术债 | ★ `docs/DESIGN.md` |
| 插件规范 | ★ `docs/PLUGIN_CHARTER.md` |
| 写第一个算子插件 | `docs/PLUGINS.md` |
| C 端插件 | `docs/PLUGIN_NATIVE.md` |
| DSL / 代码生成 | `docs/PLUGIN_DSL.md` |
| 分派与能力模型 | `docs/PLUGIN_DISPATCH.md` |
| 补后端 | `docs/BACKEND_PORTING.md`、`docs/DESIGN.md` §6 |
| 双卡验证 | `docs/DP_2GPU_RUNBOOK.md` |
| 现代 MoE | `docs/MOE_MODERN_ARCH.md` |
| 3B 性能实测 | `docs/ROADMAP_3B_PERF.md` |
| 全部文档索引 | `docs/INDEX.md` |

---

## 12. 维护本 skill 的约定

本 skill 是**长期维护件**，随项目与生态演进更新。修改时：

1. 数字必须能从 `docs/CAPABILITY_AUDIT.md` 或某个 `docs/*.md` 的实测段落复核；
2. 新增"差异"条目时，给出触发条件与处理方式，不要只写"不一样"；
3. 状态标记只用三种：**【已验证】/【已实现未验证】/【未实现】**，不得混用；
4. 每次 ORV 发布升版后，检查 §5 的规模数字与 §2 的铁律是否仍成立。
