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

一位用户在 **Tesla P100 16GB（sm_60，无 Tensor Core）** 上实测，结论要先知道，别当成普适能力：

| 项 | 事实 |
| --- | --- |
| **sm_60 本身支持** | NVRTC 能编译 `compute_60`，张量/模型/AdamW 正常训练 —— 硬件不是障碍 |
| **首装可能报 `failed to open nvrtc-builtins64_*.dll`** | **0.9.50 已修**（会扫 `nvidia/*/lib`、`*/bin` 并写进 PATH）。仍失败时的规避：把 `site-packages/torch/lib` 前置进 `PATH` **并** 调 `os.add_dll_directory` |
| **不省显存，反而多 1.2–1.9×** | 六组头对头实测一致（`docs/BENCHMARK_H2H_2026-11-12.md`）。ORV 用显存换速度与"装得下更大的模型"，**不要对外声称省显存** |
| **检查点 0.9.53 起与 PyTorch 语义对齐** | 旧版把 `_parents` 清空，导致外层图段间断开、反向递归整链重算。按 `torch/utils/checkpoint.py` 的 `CheckpointFunction` 重做后：每层反向峰值 **1.68 → 0.117 GiB**；20 层/606M/B=2/S=2048 由 **31.29 GiB 降到 7.56 GiB**（`docs/MEMORY_BACKWARD_PEAK_2026-11-12.md`）|
| **"能装下更大模型"的准确含义** | 指 PyTorch 在 **1.997B 稠密**就 OOM（31.40 GiB）而 ORV 能跑 **3B MoE**（24.78 GiB）。这是融合减少中间量，不是"更省显存" |
| **fp16 是「反向也走」但受形状门槛** | fp16 分支在**共享 GEMM 核心**里，前向反向都走；门槛 `m*n ≥ 262144` 且 `k ≤ _FP16_MAX_K`（默认不限，可用 `TG_FP16_MAX_K` 调）。**长 K 的 dW 有精度风险**：代码记录梯度相对误差会从 3e-4 恶化到 2e-2。`B=8/S=2048` 时 k=16384 正处风险区 ⇒ 开 fp16 必须在目标卡实测 |
| **AMP 不省激活显存** | 参数/激活/梯度**全程 fp32**，只有 GEMM 输入 cast。torch autocast 把激活存 fp16 ⇒ 天然省一半激活显存。**这是 ORV 显存差距的结构性原因之一**，不是"开销没优化好" |
| **`sdpa_lowmem` 只支持 `dh==64`** | `dh=128` 会**静默退回全量注意力**（不报错）。d_head=128 的模型别以为开了低显存 |
| **没有 `cos` / `sin`** | **0.9.51 已加**。旧版本需 numpy 预生成表 |
| **SDPA 维度约定** | 已接受 `is_causal=` 与 `[B,H,S,Dh]`（0.9.51 起自动展平）；旧版本要 `[B*H, S, Dh]` |

### 2.2 别把这两件事混为一谈

| | 定义 | 层级 | 状态 |
| --- | --- | --- | --- |
| **TG Link** | 用 **Vulkan 让显卡与显卡之间 P2P**，实现**多厂商交火** | 设备**之间** | 只有设计，未验证可行性 |
| **后端混用** | **一个进程里**用多种不同后端训练模型 | **进程内** | 未设计、未实现 |
| （多设备） | 同后端、多张同厂商卡、同进程 | 进程内 | 也没有（`dist.py` 是多进程） |

三条容易搞错的地方：

1. **TG Link 不是 host-staged all-reduce**。`docs/MULTIVENDOR_ROADMAP.md` §3 的 host-staged 是
   **退路**；原始设想是 Vulkan 真 P2P。真 P2P 若能成，能解锁**跨厂商 TP**；host-staged 只够 DP/PP。
2. **TG Link 与后端混用层级不同**，不能用一个方案同时解决。
3. **`_CudaAPI` 是进程级单例** —— 后端混用（进程内多后端）必须先拆它；
   而 TG Link 走跨进程，**不需要**拆它。我曾把这两者的前置条件搞反。

完整报告与复现步骤：`docs/REVIEW_P100_2026-11-12.md`。

---

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

ORV 仓库自带 TGAI GO 推理侧（`tgai.py` / `tgai.bat` + C 引擎），训完的模型可以**直接跑推理**，
不需要转成别的框架格式。

### 第 1 步：导出成 `.TG`

```python
from tgai_go.train.export import save_tg

save_tg("my_model.TG", params,          # params: {训练侧名字: numpy 数组}
        n_layers=28, d_model=1024, n_heads=16, d_ff=4096,
        vocab_tokens=tok.vocab, merges=tok.merges,
        max_seq=512, n_experts=8, n_activated=2, rope_theta=10000.0)
```

`params` 用**训练侧命名**（`experts.{e}.gate/up/down`）；导出时会映射成引擎的 `w1/w2/w3`。

### 导出时三个会让模型"能加载但生成崩"的坑

| 坑 | 症状 | 处理 |
| --- | --- | --- |
| **RoPE 的 cos/sin 没填** | 加载正常，`generate` 时 `access violation reading 0x…` —— 引擎**按指针直接读**这块 buffer | `collect()` 已按公式补齐，别绕过它自己拼 cfg |
| **`head_dim != d_model // n_heads`** | **导出必错且不报错**，引擎按 `d_model//n_heads` 解析 | `.TG` 头里**没有 head_dim 字段**，表达不了这种结构。要么改结构，要么换格式 |
| **RoPE theta 与训练侧不一致** | 位置编码错位，输出乱 | 训练用什么 theta，导出就传什么 |

### 第 2 步：用 GO 跑起来

```bash
tgai.bat models                          # 列出可用模型（序号 + 大小）
tgai.bat chat -m my_model.TG             # 直接对话
tgai.bat server -m my_model.TG           # 只开 OpenAI 兼容服务（默认 8080）
tgai.bat server -m my_model.TG --port 11435
tgai.bat chat --server 8080              # 对话 + 同进程开服务
tgai.bat bench                           # 基准测试
```

选择计算后端（加 `--dll`）：

```bash
tgai.bat chat --dll cpu\build\libtgai_engine.dll              # 纯 CPU
tgai.bat chat --dll cpu\build\libtgai_engine_gpubridge.dll    # GPU 桥
```

对话内命令：`/help` `/params` `/model <路径>` `/server [端口]` `/env` `/stats` `/clear` `/exit`

### 已有 `.pt` / `.gguf` 想转 `.TG`

```bash
tgai.bat convert <args>
```

细节见 `docs/GGUF_TO_TG.md`、`docs/TG_V4_FORMAT.md`。

### 交付前必须验证的三件事

1. **加载不崩**：`tgai.bat models` 能列出来
2. **生成不崩**：问一句话能出完整回复（RoPE buffer 缺失在这一步才暴露）
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
| **想混用不同后端 / 混杂厂商多卡（TG Link）** | ★ `docs/MULTIVENDOR_ROADMAP.md`（**权威**，别从零推导） |
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
