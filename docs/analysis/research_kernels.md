# 训练引擎 Kernel 加速：可落地技术调研与排序（2024–2026）

> 目标引擎：纯 Python + ctypes → CUDA Driver API，内核为 NVRTC 运行时编译的源码字符串，无 PyTorch。
> 目标硬件：Tesla V100，`sm_70`，fp32 计算（bf16 仅存储）。
> 实测基线（B=4 / S=512 / Qwen2.5-0.5B）：forward 0.419 s，backward 0.723 s，opt 0.024 s，合计 **1.166 s**；PyTorch 同形状 **0.677 s**。
> 本文所有"我们的估算"都用 Qwen2.5-0.5B 的**真实 config**（hidden 896 / 24 层 / 14 Q heads / 2 KV heads / intermediate 4864 / vocab 151936 / tied embedding）计算，配置来源：[Qwen2.5-0.5B config.json](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json)。

---

## 0. 结论先行（TL;DR）

1. **FA2/FA3/FA4 的"算法层"技巧（不物化 P、recompute、分块 online softmax、warp 级 reduce、swizzle）在 fp32 + sm_70 上全部有效；"硬件层"技巧（TMA、wgmma、warp-specialization、FP8、TMEM、cp.async）全部无效**——后者有 PTX ISA 的硬性 `Requires sm_90a / sm_80` 判据，不是"调优问题"。
2. **但按你们的形状算，attention 核心只占 forward FLOPs 的 4.3%**，把所有 attention 中间张量开销（mask、permute、softmax 两趟、KV×7）全部归零，**收益上界约 15–45 ms（1–4%）**。这不足以解释 0.49 s 的差距。
3. **真正的 0.49 s 差距最可能来自"GEMM 没跑满"而不是"非 GEMM 太多"**：总 GEMM 下界 0.56–0.58 s，实测 1.166 s，二者之差 0.59 s；而全部 elementwise + attention 中间件的带宽账只有 25–60 ms。**先测分 shape 的 GEMM 效率，再决定改什么**（第 6 节 P0）。
4. **Liger / Cut-CE 的 fused linear CE 对你们很可能是"负时间收益"**：它省 1.2–3.7 GB 显存，但 backward 要多算一次 lm_head GEMM（约 +51 ms），只有 OOM 时才该上。这与 Liger 自己在 B200 上公布的 backward 变慢数据一致。
5. **视图语义是唯一"低风险、零数值风险、必做"的一项**：kernel 加 stride 参数 + GQA 用 `h // group` 索引，可一次性消灭 `permute` / `repeat_interleave` 的全部真拷贝，且**不需要改 autograd 契约**（第 5 节给出等价性证明与代码骨架）。

---

## 1. 先把账算清楚：0.585 s 的"非 GEMM"到底在哪

### 1.1 FLOPs 分解（forward，B=4 × S=512 = 2048 tokens）

| 组成 | 公式 | FLOPs | 占比 |
|---|---|---:|---:|
| MLP（gate/up/down） | 3 × 2·h·inter·tokens × 24 | 1285 GFLOP | 60.8% |
| lm_head（tied，仍要算） | 2·h·V·tokens | 558 GFLOP | 26.4% |
| Attention 投影（q/k/v/o） | 2·(h² + 2·h·d_kv + h²)·tokens × 24 | 181 GFLOP | 8.6% |
| **Attention 核心（QKᵀ + PV）** | 4·B·H·S²·D × 24 | **90 GFLOP** | **4.3%** |
| 合计 | | **2113 GFLOP** | 100% |

- backward ≈ 2 × forward（attention 部分按 FA2 的定价是 **2.5×**，因其 backward 有 5 个 matmul 而 fwd 只有 2 个，但 attention 占比小，整体仍近似 2×）→ 总训练 FLOPs ≈ **6.34 TFLOP**。
- 以你们实测 GEMM 10.9 TFLOPS 折算，**纯 GEMM 下界 = 0.58 s**。
- 实测 1.166 s ⇒ **"非 GEMM + GEMM 效率损失"合计 0.585 s**。
- PyTorch 0.677 s ⇒ 它的同一项只有 **0.10 s**。

结论：**差距 0.47–0.49 s 全部落在这 0.585 s 里**。下面逐项核对这 0.585 s 能被什么解释。

### 1.2 各项"非 GEMM"的带宽账（V100 HBM ≈ 900 GB/s，理论值）

单层关键中间张量（fp32，B=4/H=14/S=512/D=64）：

- `scores` = 4×14×512×512×4 B = **58.7 MB**
- `mask [S,S]` 广播 = 1 MB（若物化成 `[B,H,S,S]` 则是 58.7 MB）
- K/V 单层 = 2 × 4×2×512×64×4 B = **2.1 MB**；放大 7 倍后 14.7 MB（增量 12.6 MB）

按"每层物化 6 次读写 scores"估算：

| 项 | 每层流量 | 24 层 | @900 GB/s |
|---|---:|---:|---:|
| scores 写 + mask 加（读+写） | 176 MB | 4.2 GB | 4.7 ms |
| softmax 两趟（读 + 读 + 写） | 176 MB | 4.2 GB | 4.7 ms |
| backward 读 P + 写 dP + 读 dP | 176 MB | 4.2 GB | 4.7 ms |
| GQA K/V 放大 7×（fwd+bwd） | 25 MB | 0.6 GB | 0.7 ms |
| **合计** | | **13.2 GB** | **≈ 15 ms** |

即使按"有效带宽只有峰值 1/3"这种最坏情况，也只有 **≈ 45 ms**。
再加上全部 elementwise（RMSNorm / SiLU / residual / RoPE，每层约 20–30 个 kernel，共约 700 个 kernel，每个 15–70 µs）≈ **15–50 ms**。

**⇒ attention 中间件 + elementwise 的天花板约 60–95 ms，仍差 0.49 s 一个数量级。**

### 1.3 那 0.49 s 最可能是什么（按可疑度排序）

| 假设 | 机制 | 可回收量级 | 验证方法 |
|---|---|---:|---|
| **H1：GEMM 分 shape 效率不均** | 你们测的 10.9 TFLOPS 可能只是某个理想 shape（如 MLP fwd）。backward 的 dW GEMM 是 M/N 小、K 大的形状，tile 数 < SM 数时 occupancy 不足 | **100–300 ms** | 按 (M,N,K) 分组，逐组测 TFLOPS |
| **H2：显存分配 / 隐式同步** | 每次 forward 为 24 层分配 `scores`（58.7 MB×24）等临时张量；`cudaMalloc` 是**全局同步**操作，PyTorch 用 caching allocator 专门规避 | 50–200 ms | `nsys` 统计 `cudaMalloc` 次数与时间线空洞 |
| **H3：kernel launch / CPU 侧瓶颈** | 约 1500–2500 次 launch，ctypes 每次有 Python→C 开销；若每层之间有 `cudaDeviceSynchronize` 或 D2H 读 loss，单次 100–300 µs | 30–150 ms | `nsys` 看 GPU idle gap 占比 |
| **H4：elementwise 未向量化** | 未用 `float4`、未合并访存、grid 配置差 → 有效带宽 200–300 GB/s | 30–80 ms | 单 kernel 微基准 vs `cudaMemcpy` 带宽 |
| **H5：attention 中间件** | 见 §1.2 | **10–45 ms** | 单层 attention 隔离计时 |

> **这是本文最重要的判断**：在没做 H1–H3 之前重写 attention，最多拿到 1–4% 的端到端收益。
> FA 式重写的真正价值是**为 S 变长铺路**（S=4096 时 P = 3.76 GB/层，物化直接不可行），不是解决当前 0.49 s。

---

## 2. Q1：FlashAttention 2/3/4 的哪些技巧在 fp32 + sm_70 上仍然有效

### 2.1 硬件事实（有 PTX ISA 原文判据，不是经验之谈）

| 特性 | 最低计算能力 | V100 (sm_70) | 判据来源 |
|---|---|---|---|
| `mma.sync.m8n8k4` `.f16`（V100 的 HMMA） | `sm_70` | ✅ 可用，且**专为 sm_70 优化** | [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)：*"`.f16` floating point type mma operation with `.m8n8k4` shape requires sm_70 or higher. Note mma.sync.m8n8k4 is optimized for target architecture sm_70"* |
| `mma` `.bf16` / `.bf16x2` packed / `.tf32` | `sm_80` | ❌ 无 | 同上：*"`.bf16`, `.bf16x2`, `.tf32` destination formats require sm_80"*；`fma{.relu}.{bf16,bf16x2}` 亦 *"require sm_80"* |
| `cp.async`（异步 global→shared） | `sm_80` | ❌ 无 | 同上：*"Target ISA Notes: Requires sm_80 or higher"*（cp.async 章节） |
| `wgmma.mma_async`（Hopper warpgroup MMA） | `sm_90a` | ❌ 无 | 同上：*"Target ISA Notes: Requires sm_90a"* |
| TMA / warp-specialization / FP8 block-quant | Hopper (`sm_90`) | ❌ 无 | [FA3 论文](https://arxiv.org/abs/2407.08608) 明确把这三者列为 Hopper 专属 |
| TMEM（Tensor Memory） | Blackwell | ❌ 无 | [PyTorch blog: FlexAttention + FA4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)（2026-03-05） |

**两条直接推论（对你们影响最大）**：

- **"bf16 存储可做、bf16 计算不行"在 V100 上不是软件选择，是硬件事实**。V100 的 Tensor Core 只吃 fp16；bf16 的 MMA 要 sm_80。所以 bf16 存储只能省带宽/显存，**不会加速 GEMM**；要上 Tensor Core 必须走 **fp16 + fp32 累加 + loss scaling**（V100 fp16 TC 的理论峰值远高于 fp32 CUDA core，但溢出风险与数值验证成本要另算）。
- **FA2 的软件流水线（`cp.async` 双缓冲 + 多 stage）在 sm_70 上根本无法编译**。V100 只能做同步 `ld.global → st.shared`，或手工双缓冲（用普通 load 提前发射）。这意味着**即便照抄 FA2 的 C++ 结构，也拿不到它在 A100 上的那部分收益**。

### 2.2 逐技巧判定（附定量判据）

| 技巧 | 依赖 | V100 fp32 判定 | 判据（可算的数） |
|---|---|---|---|
| **不物化 P**（tiling + 在 SRAM 内完成 softmax·V） | 纯算法 | ✅ **有效** | 省 `k·B·H·S²·4B` 流量。我们 S=512：单层 58.7 MB，k≈3 ⇒ 176 MB/层 ⇒ 24 层 4.2 GB ⇒ **4.7 ms**。fp32 是 fp16 的 2 倍字节，**相对收益比 fp16 更大** |
| **分块 online softmax**（running max/sum + rescale） | 纯算法 | ✅ **有效** | fp32 尾数 24 bit，rescale 的相对误差 ~1e-7，可忽略；与 fp16 不同，**不需要**为数值稳定性额外补偿 |
| **recompute**（backward 重算 S 而非存 P） | 纯算法 | ✅ **有效** | 省 `B·H·S²·4B = 58.7 MB/层`（1.41 GB 总量）；代价 = 多 1 次 QKᵀ GEMM（1.88 GFLOP/层 ⇒ 0.17 ms/层 ⇒ **4.1 ms 总量**）。FA2 原文把 attention backward 定为 **fwd 的 2.5× matmul**（`5 matmuls vs 2, due to recomputation`）。**用 4 ms 换 1.4 GB，非常划算** |
| **warp 级 row reduce**（`__shfl_xor_sync` 求 row max/sum） | sm_70 原生 | ✅ **有效** | 替代 "shared memory + `__syncthreads()`"，每块省 2–4 次 barrier；V100 shuffle 为 1 inst/cycle/warp，全速 |
| **shared memory swizzle / padding** | 纯实现 | ✅ **有效，且比 fp16 更关键** | V100 32 banks × 4 B。fp16 时一个 bank 装 2 个元素，冲突天然减半；**fp32 时 1 bank = 1 元素，列访问必然 32-way 冲突**，必须 padding 或 XOR swizzle |
| **减少 non-matmul FLOPs**（FA2 §3.1：只存 logsumexp、延后除法） | 纯算法 | ⚠️ **有效但收益远小于 A100** | FA2 的论据是 A100 上 *"312 TFLOPs/s FP16 matmul, but only 19.5 TFLOPs/s non-matmul FP32"*（16×）。**V100 fp32 上 matmul 与 non-matmul 都是 CUDA core，比例≈1×** ⇒ 这条优化的杠杆在你们这里基本消失 |
| **沿 seqlen 切 CTA 提 occupancy** | 纯算法 | ⚠️ **条件有效** | 判据：`grid = B·H·ceil(S/BM)`。我们 `B·H = 56 < 80 SM` ⇒ **有效**（能提到 112/168）。若 `B·H ≥ SM 数` 则无收益 |
| **warp 内切 Q 行**（FA2 的 4-warp 分工，避免 shared 通信） | 纯实现 | ✅ **有效** | 纯寄存器 + shuffle，无额外硬件依赖 |
| `cp.async` 多级流水 | sm_80 | ❌ **不可用** | PTX：Requires sm_80 |
| TMA + warp-specialization | sm_90 | ❌ **不可用** | FA3 论文 |
| `wgmma` | sm_90a | ❌ **不可用** | PTX：Requires sm_90a |
| FP8 block quantization | sm_89+ | ❌ **不可用** | FA3 论文 |
| TMEM / 全异步 pipeline | Blackwell | ❌ **不可用** | PyTorch FA4 blog |

### 2.3 FA2 vs FA3 vs FA4：对你们各自意味着什么

- **FA2**（[arXiv 2307.08691](https://arxiv.org/abs/2307.08691)）：三项贡献是 (1) 减少 non-matmul FLOPs、(2) 沿 seqlen 并行、(3) warp 间分工减少 shared 通信。在 A100 上比 FA1 快约 2×，达 50–73% 峰值；**forward 30–50%、backward 25–35%** 峰值利用率。→ 对你们：**(2)(3) 可搬，(1) 杠杆很弱**。
- **FA3**（[arXiv 2407.08608](https://arxiv.org/abs/2407.08608)）：H100 上 1.5–2.0×，FP16 达 740 TFLOPS（75%），FP8 近 1.2 PFLOPS。三项技术全部建立在 **TMA + wgmma 异步性**上。→ **对 V100 零可搬性**，只能借鉴"softmax 与 GEMM 交错"这一思想（在 CUDA core 上即"边算边 rescale"，本来 FA2 就有）。
- **FA4**（[PyTorch blog, 2026-03](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)）：CuTeDSL 实现，只针对 Hopper/Blackwell；官方 README 写明 *"FlashAttention-4 is written in CuTeDSL and optimized for Hopper and Blackwell GPUs"*（[flash-attention README](https://github.com/Dao-AILab/flash-attention)）。→ **不可搬**。
- **官方支持矩阵的硬事实**：FA2 的 CUDA 实现只支持 *"Ampere, Ada, or Hopper"*，且 *"Datatype fp16 and bf16"*；**Volta 与 fp32 都不在列**。唯一支持 fp32 的是 **Triton 实现**（*"supports ... fp16, bf16, and fp32 datatypes"*，含 MQA/GQA、causal、任意 head dim）。→ 所以"fp32 的 FA"在工程上是存在的（Triton 路径），但**在 V100 上跑 Triton 的 fp32 FA 是否会快过物化实现，必须实测**，不能假设。

> **一句话判据**：FA 的加速 = ΔHBM流量 / 带宽 + Δoccupancy 收益。**它从来不是"用了 tensor core 所以快"**。在 fp32 + sm_70 上，你能拿到的只有前两项；A100 上那 2–4× 里的大头（Tensor Core 峰值 + async pipeline）你一分都拿不到。**预期收益应按 1.2–1.5× 估，不是 2–4×**。

---

## 3. Q2：Liger / Unsloth / Fused CE 一类"训练侧融合"

### 3.1 各家报告的数字（原文）

| 工作 | 报告数字 | 来源 |
|---|---|---|
| **Liger-Kernel** | 平均 **+20% 训练吞吐**、**−60% GPU 显存**（对比 HuggingFace 实现）；HF Trainer 例子：LLaMA3-8B **快约 20%、省 40%+ 显存**（4×A100 FSDP）；Medusa 例：**省 80% 显存、+40% 吞吐** | [论文 arXiv 2410.10989](https://arxiv.org/abs/2410.10989)、[README](https://github.com/linkedin/Liger-Kernel) |
| **Cut Cross-Entropy (CCE)** | Gemma 2 (2B)：loss 计算显存 **24 GB → 1 MB**；分类头总显存 **28 GB → 1 GB**；**不牺牲训练速度或收敛** | [arXiv 2411.09009](https://arxiv.org/abs/2411.09009) |
| **Unsloth** | *"2× faster with 70% less VRAM"*；分模型表：Llama 3.1 8B 2× / −70%，gpt-oss 20B 2× / −70% | [README](https://github.com/unslothai/unsloth) |
| **FlexAttention** | 可表达多数 attention 变体，性能与手写 kernel 竞争 | [arXiv 2412.05496](https://arxiv.org/abs/2412.05496) |

> ⚠️ **归因警告**：Unsloth 的 2×/70% 是**端到端 fine-tuning 数字**，其中包含 4-bit 量化、手动 autograd、gradient checkpointing 等，**不能归因于 kernel fusion**。Liger 的 20%/60% 也是"vs HuggingFace 实现"的整体数字，含"消除中间张量 + chunking + in-place"。**把这些数字直接套到自研引擎上会严重高估**。

### 3.2 对你们的逐项估算（Qwen2.5-0.5B，2048 tokens）

| 技术 | 报告收益类型 | 我们的显存收益 | 我们的时间收益 | 判定 |
|---|---|---:|---:|---|
| **Fused linear CE（Liger FLCE / CCE）** | 显存为主 | **省 1.24–3.7 GB**（logits `[2048,151936]` fp32 = 1.24 GB；若同时存 probs 则 2.48 GB；再加 backward 的 dlogits 峰值 ≈ 3.7 GB） | **≈ +51 ms（负收益）** | ⚠️ **只在 OOM 时启用** |
| **Fused RMSNorm** | 时间/显存 | 可忽略（7.3 MB 级） | −1～−3 ms（省 1 次读写 + 1 次 launch/层） | ✅ 低风险可做 |
| **Fused RoPE** | 时间 | 可忽略 | −1～−3 ms（我们 `fused_rope` 已写好但未接线，**先接线**） | ✅ 立刻做 |
| **Fused SwiGLU** | 时间 | 可忽略 | **已做**（`silu_mul` fwd/bwd 融合） | ✅ 已完成 |
| **Chunked CE** | 显存 | 同 FLCE（分块后峰值 = chunk×V×4） | 同 FLCE | 同 FLCE |

### 3.3 为什么 FLCE 对你们是负收益（这是本文最反直觉、也最该看的一条）

FLCE 的核心是"**不物化 logits，在 SRAM 里边算边做 log-sum-exp**"。它省的是显存和 HBM 流量，代价是 **backward 必须重算 logits** 才能得到 `dlogits = (softmax(logits) − onehot) / N`。

算一下你们的账：

- **现状（物化）**：backward 直接读已存的 probs（1.24 GB）⇒ `1.24 GB / 900 GB/s ≈ 1.4 ms`。
- **FLCE（重算）**：backward 要重做一次 `[2048,896] × [896,151936]` 的 GEMM ⇒ `558 GFLOP / 10.9 TFLOPS ≈ 51 ms`。
- **净差 ≈ +50 ms**，换来省 1.2–3.7 GB 显存。

Liger 自己在 B200 上的公开数据也印证这一点（[README](https://github.com/linkedin/Liger-Kernel) 的 `Forward | Backward | Full | Peak full memory` 表）：

| 实现 | Forward | Backward | Full | 峰值显存 |
|---|---:|---:|---:|---:|
| cuTile（FLCE） | **2.97 ms** | 8.35 ms | **11.36 ms** | **2.64 GiB** |
| Torch（物化） | 5.24 ms | **7.15 ms** | 12.85 ms | 7.22 GiB |

⇒ **FLCE 的 forward 快 1.76×，backward 反而慢 17%**，只有 "Full" 因为 forward 的收益才略快；**显存从 7.22 → 2.64 GiB 才是它的主卖点**。

**判据**：`是否启用 FLCE = (当前峰值显存 > 0.85 × 卡容量) ? 启用 : 不启用`。
你们的 backward 已经是瓶颈（0.723 s），**在没有 OOM 的前提下上 FLCE 会让 backward 更慢**。

---

## 4. Q3：backward 专项——哪些工作真的优化了反向

### 4.1 FA2 的 dQ atomic 真相（纠正一个常见误传）

FA2 论文原文（[arXiv 2307.08691](https://arxiv.org/abs/2307.08691)，§3.3 并行化）：

> *"We thus parallelize over the sequence length dimension as well, and schedule 1 thread block for each column block of the backward pass. **We use atomic adds to communicate between different thread blocks to update dQ.**"*

**FA2 没有消除 atomic，它是主动引入 atomic 来换取 backward 的 seqlen 并行度**。真正"无 atomic"的条件是：

> **每个 CTA 拥有完整的 Q 行块、遍历全部 K/V**（即 backward 不沿 seqlen 切分），此时 `dQ` 的每一行只被一个 CTA 写，天然无需 atomic。

对我们的判据：

| 方案 | atomic 次数 | occupancy | 适用条件 |
|---|---:|---|---|
| 不切 seqlen（每 CTA 一行块 Q，遍历全 K/V） | **0** | `B·H = 56`（< 80 SM，偏低） | S 较短、`B·H` 接近 SM 数 |
| 切 seqlen 为 `n` 段 + atomic dQ | `n × B·H·S·D` 次 atomic 写 | `56·n` | `B·H ≪ SM` 且 S 长 |
| 切 seqlen + 两阶段 reduce（无 atomic，确定性） | 0（但多一次 kernel + `n×` 中间缓冲） | 同切分 | 需要**逐位可复现**时 |

> **关键**：atomic 本身不是问题，**atomic 冲突**才是。dQ 按 `[B,H,S,D]` 排布时，同一 cacheline（32 B = 8 个 fp32）会被 8 个不同 `d` 的线程写；若再叠加多个 seqlen 段的 CTA 同时写同一行，L2 上的 atomic 串行化会非常严重。**V100 的 atomic 走 L2，冲突时是串行化的**，这一点在 fp32（4 B/元素）下比 fp16 更糟。

### 4.2 GQA 的 `repeat_interleave` 消除（收益确定、风险最低）

PyTorch 官方文档给出的 SDPA **数学等价参考实现**里明确写着（[SDPA docs](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)）：

```python
if enable_gqa:
    key   = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
    value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
```

**这是"语义定义"，不是"高效实现"**。PyTorch 自己的 issue 就记录了这个坑：当走广播（不拷贝）时不选中高效 kernel、显存飙升；而 `enable_gqa=True` 的文档实现又是真拷贝 —— 见 [pytorch#154363 "scaled_dot_product_attention broadcasting (GQA) is a memory footgun"](https://github.com/pytorch/pytorch/issues/154363)。

**正确做法**：kernel 内用 `h_kv = h // group`（`repeat_interleave(group, -3)` 的精确语义就是"每 group 个连续 head 共享同一个 KV head"），K/V 指针按 `h_kv` 偏移，**零拷贝**。

我们的量级：group = 14/2 = **7**；单层 K/V 原始 2.1 MB → 放大后 14.7 MB（增量 12.6 MB），fwd+bwd 约 25 MB/层，24 层 ≈ 0.6 GB ≈ **0.7 ms**。
⇒ **时间收益很小（<1%），但它同时消掉 2 个 kernel 和 1 份 14.7 MB×24 的临时缓冲，且是后续所有 attention 改造的前置条件。属于"必做但不指望它救命"**。

### 4.3 避免物化 mask 的 causal softmax

三种做法，代价差异巨大：

| 做法 | 每层额外流量 | 判定 |
|---|---:|---|
| 物化 `[B,H,S,S]` 的加性 mask 再 `scores += mask` | 写 58.7 + 读 58.7 MB | ❌ 最差 |
| 广播 `[S,S]` mask 加到 `[B,H,S,S]` | 读 58.7 + 写 58.7 MB（mask 只读 1 MB） | ⚠️ 可接受 |
| **用 `is_causal` 标志 + 按行范围循环**（只算 `j ≤ i` 的块，`-inf` 块直接跳过） | **0**，且 exp 量减半 | ✅ 最优 |

第三种还能**顺带省掉一半的 exp 和一半的 scores 读**：causal 下有效元素只有 `S(S+1)/2 ≈ S²/2`。

FA2 论文对这条给了直接数据（§3.2，原文）：跳过"行索引严格小于列索引"的块、且每行只对 1 个对角块施加 mask，**"leads to around 1.7–1.8× speedup compared to attention without the causal mask"**。这是 A100 fp16 的数字，fp32 下收益结构相同（省的是同一批非 matmul 操作与流量），可类比。

我们的量级：省 88 MB/层 ⇒ 2.1 GB ⇒ **2.3 ms**（时间绝对值小，因为 S=512 时 attention 本身只占 4.3%），但**省掉的是"每次都要读一遍 mask 再写一遍 scores"的确定性开销**，且是 correctness 更容易保证的写法（不用生成 `-inf` 再相加）。

### 4.4 我们真正的 backward 大头：dW GEMM 的形状

这条不在论文里，但按形状算最可疑（对应 §1.3 的 H1）：

| GEMM | (M, N, K) | 128×128 tile 的 tile 数 | occupancy (80 SM) |
|---|---|---:|---:|
| MLP gate/up (fwd) | (2048, 9728, 896) | 16 × 76 = 1216 | 满 |
| MLP down (fwd) | (2048, 896, 4864) | 16 × 7 = 112 | 满（刚过线） |
| lm_head (fwd) | (2048, 151936, 896) | 16 × 1187 | 满 |
| **q/k/v/o 的 dW (bwd)** | **(896, 896, 2048)** | **7 × 7 = 49** | **61%（不足）** |
| **MLP down 的 dW (bwd)** | **(896, 4864, 2048)** | **7 × 38 = 266** | 满 |

`dW` 类 GEMM 的共同特征是 **M/N 小、K 大** ⇒ 天然适合 **split-K / stream-K**：

- **Split-K**：把 K=2048 切成 4 段，每段一个 CTA，最后 reduce ⇒ tile 数 ×4，occupancy 拉满；代价是多一次 reduce（`4 × 896×896×4B = 12.9 MB`，可忽略）。
- **Stream-K**（[ACM PPoPP'23, DOI 10.1145/3572848.3577479](https://doi.org/10.1145/3572848.3577479)）：work-centric 分解，把"固定 tile 数"改成"固定 CTA 数、均分 MAC 迭代"，专治**尾效应（tail effect）**与低 occupancy；CUTLASS 已内置该调度。

**预期收益**：若 q/k/v/o 的 dW 从 61% occupancy 恢复到 ~95%，这部分 FLOPs 是 `2 × 1.835M × 2048 × 24 = 180 GFLOP`（占总 2.8%），即便提速 1.5× 也只省 **~7 ms**。
⇒ **单看这一项不大**。但如果你们的 GEMM 在**多个** shape 上都只有 6–8 TFLOPS（而不是 10.9），那问题就是全局的 tile/split 策略，收益会到 **100–300 ms**。**必须先测分 shape 表**（§6 P0）。

### 4.5 recompute 的取舍（唯一"确定赚"的 backward 优化）

| 方案 | 显存 | 时间 |
|---|---:|---:|
| 存 P（`[B,H,S,S]` fp32） | 58.7 MB/层 ⇒ **1.41 GB** | 0（直接读） |
| 重算 S（存 Q/K/V + L，backward 重做 QKᵀ） | ~0 | +1.88 GFLOP/层 ⇒ **+4.1 ms** 总 |

**用 4.1 ms 换 1.41 GB**——这是全文性价比最高的一笔交易。FA2 的做法继承自 FA1：论文 §3.1.2 明确 *"The backward pass of FlashAttention-2 is almost the same as that of FlashAttention"*，并在 §3.2 给出代价的原文定价——*"we multiply the forward pass FLOPs by 2.5 (since there are 2 matmuls in the forward pass and 5 matmuls in the backward pass, due to recomputation)"*。**即 attention backward 的 matmul 数是 forward 的 2.5 倍，这个 2.5× 就是"重算"的价格标签**（纯非 attention 部分仍是 2×）。

---

## 5. Q4：视图语义——kernel 层如何避免拷贝，最小改动是什么

### 5.1 PyTorch 的 view 到底是什么

`permute` / `transpose` / `expand` / `reshape`（当可视图时）**只改 stride 元组，不搬数据**；`expand` 更极端，把对应维的 stride 设为 **0**，于是"7 个 head"共享同一块内存。**没有任何魔法**：卷积/GEMM/attention kernel 本来就按 stride 寻址，所以"视图"对 kernel 而言只是"换了一组 stride 参数"。

我们的引擎之所以被迫真拷贝，是因为 kernel 的地址计算**硬编码了连续布局**（`idx = ((b*H + h)*S + s)*D + d`），一旦上游是 `permute` 过的张量，就必须先 `contiguous()`。

### 5.2 最小改动（4 步，不动 autograd 契约）

**第 1 步：所有 attention 相关 kernel 增加 4 个 stride 参数**（用 `int64_t`，避免大张量溢出）：

```c
// 每个张量 4 个 stride，单位=元素（不是字节）
__global__ void attn_fwd(
    const float* __restrict__ Q, int sqb, int sqh, int sqs, int sqd,
    const float* __restrict__ K, int skb, int skh, int sks, int skd,
    const float* __restrict__ V, int svb, int svh, int svs, int svd,
    float*       __restrict__ O, int sob, int soh, int sos, int sod,
    int B, int H, int HKV, int S, int D, float scale, int is_causal)
{
    const int hkv = blockIdx.y;            // KV head
    const int hq  = blockIdx.z;            // Q head
    // GQA：Q head hq 对应 KV head = hq / (H / HKV)，等价于 repeat_interleave(group, -3)
    const int group = H / HKV;
    const int kh = hq / group;
    // 地址 = 基址 + b*sb + h*sh + s*ss + d*sd  （sd 可为 1，也可不为 1）
}
```

**第 2 步：把 `repeat_interleave` 调用点直接删掉**，改为传 `HKV=2`（而不是 `H=14`）+ 原始的 K/V 指针。Q 侧不需要任何变换。

**第 3 步：`permute` 一律不 materialize**。上游若产出 `[B,S,H,D]`（seq-major），就传 `sh = D, ss = H*D`；若产出 `[B,H,S,D]`，就传 `sh = S*D, ss = D`。**同一份 kernel 二进制，靠参数区分**。

**第 4 步：backward 的 K/V 梯度用 `index_add` 语义归约**，这是唯一需要新写的部分：

```c
// dK/dV: [B, HKV, S, D]；每个 (b, kh, s, d) 累加所有 hq ∈ [kh*group, (kh+1)*group)
// 用 atomicAdd 到 dK（因为 7 个 Q head 会写同一位置），或先写 [group] 份再 reduce
atomicAdd(&dK[b*skb + kh*skh + s*sks + d*skd], dk_local);
```

> 注意：这里**必须用 atomic**（或分组 reduce），因为 GQA 的梯度天然是"7 对 1"的归约。但这只发生在 **dK/dV** 上（大小 2.1 MB/层），不在 dQ 上，冲突面很小。

### 5.3 为什么"不改 autograd 契约"

autograd 契约只要求两件事：
1. **forward 的输出形状/数值语义不变**：输出仍是 `[B,H,S,D]`（或你们当前的布局），stride 变化对 `.shape` 无影响。
2. **backward 返回的梯度形状与输入一致，且满足链式法则**：`repeat_interleave` 的伴随算子是 `index_add`（把 group 个梯度加回去）——这正是第 4 步做的。数学上 `dL/dK = Σ_{h∈group} dL/dK_expanded[h]`，与 expand-then-reduce 完全等价。

**唯一的语义变化点**：如果你们现在依赖"K/V 是连续且被复制过的"来做某些 in-place 操作，需要改成"只读 + 索引"。这是实现约束，不是契约变化。

### 5.4 验收判据（可证伪）

1. **数值**：与当前实现逐元素比对，`max_abs_err < 1e-6`（fp32），且 backward 的 `dK/dV` 在 `atomicAdd` 非确定性下允许 `1e-5` 相对误差（或改用确定性两阶段 reduce 后要求 `1e-6`）。
2. **零拷贝证据**：`nsys` 中 kernel 名含 `copy` / `elementwise` 的条目数从 N 降到 0；或用 `cudaMemcpyAsync` 计数为 0。
3. **性能**：单层 attention fwd 时间下降 ≥ 15%（主要来自去掉 2 次 58.7 MB 级拷贝 + 2 次 launch）。
4. **正确性回归**：`is_causal=0/1`、`HKV ∈ {2, 14}`、`S ∈ {1, 7, 512, 513}`（非 2 的幂）四组交叉测试全过。

---

## 6. Q5：排序后的落地清单（10 项）

> 排序原则：**先测量（P0）→ 再消除系统性开销（P1–P2）→ 再算法优化（P3–P5）→ 最后精度路线（P6）**。
> "预期收益"是**端到端 1.166 s 上的绝对毫秒数**，不是单项加速比。

### P0 —— 分 shape GEMM 效率表 + `nsys` 全局账（**不做这个，后面全是猜**）

- **技术名**：按 `(M,N,K)` 分组测量 GEMM 实际 TFLOPS；用 `nsys` 统计 kernel 数、`cudaMalloc` 次数、GPU idle gap。
- **来源**：[NVIDIA CUDA Graph 量化收益方法论](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/quantitative-benefits.html)（用 `nsys` 数 launch 数与利用率的标准做法）。
- **预期收益**：**0 ms 直接**，但决定后面 300 ms 的方向。
- **难度**：**低**（1–2 天）。
- **V100 适用**：✅。
- **验收判据**：产出一张表，列出每个 GEMM 的 `(M,N,K)`、耗时、TFLOPS；标出所有 `< 8 TFLOPS` 的项。若"GEMM 总时间" < 0.58 s，说明账算错了，重测。

### P1 —— 显存池化 + CUDA Graph（消除 `cudaMalloc` 同步与 launch 开销）

- **技术名**：自建 caching allocator（按 size bucket 复用）+ `cuStreamBeginCapture` / `cuGraphLaunch` 捕获整步训练。
- **来源**：[NVIDIA CUDA Graph 文档](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/quantitative-benefits.html)（明确写 *"By eliminating CPU-side launch overhead..."*、*"significantly reduce the driver-side and hardware-side overhead associated with kernel launches"*，并给出 "without graphs 78% → with graphs 95%" 的 SM 利用率示例）。
- **预期收益**：**30–150 ms**（若 H2/H3 成立）。
- **难度**：**中**（CUDA Graph 要求整图静态、无 `cudaMalloc`、无 D2H 同步；需要先把 allocator 做成图安全的）。
- **V100 适用**：✅（Graph 是 CUDA 10+ 特性，与 CC 无关）。
- **验收判据**：`nsys` 中 `cudaMalloc` 调用数 = 0（稳态）；GPU idle gap 占比从 X% 降到 <5%；端到端时间下降 ≥ 3%。

### P2 —— Stride 化 attention + GQA 索引（消灭 `permute` / `repeat_interleave`）

- **技术名**：kernel 传 stride，GQA 用 `h // group`。
- **来源**：[PyTorch SDPA 文档的等价实现](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)、[pytorch#154363](https://github.com/pytorch/pytorch/issues/154363)。
- **预期收益**：**5–15 ms** + 省 0.6 GB 临时缓冲。
- **难度**：**低**（第 5 节给了骨架）。
- **V100 适用**：✅。
- **验收判据**：§5.4 四条。

### P3 —— causal mask 标志化（不物化 `[S,S]` mask）

- **技术名**：`is_causal` 标志 + 按行范围分块，`-inf` 块直接跳过。
- **来源**：同上 SDPA 文档（`is_causal` 语义）；FA2 论文的分块思想。
- **预期收益**：**2–8 ms**（省 88 MB/层流量 + 一半 exp）。
- **难度**：**低**。
- **V100 适用**：✅。
- **验收判据**：`scores` 相关 kernel 的 HBM 读写量下降 50%（用 `ncu` 的 `dram__bytes` 或 nsys 估算）；数值与"物化 mask"版本 `max_abs_err < 1e-6`。

### P4 —— attention backward 的 recompute（不存 P）

- **技术名**：forward 只存 `Q,K,V,O` 与 `logsumexp L`（`B·H·S` 个 fp32），backward 重算 `S = QKᵀ`。
- **来源**：[FA2 论文](https://arxiv.org/abs/2307.08691) §2.3（*"we recompute the attention matrix during the backward pass"*）。
- **预期收益**：**省 1.41 GB 显存**，代价 +4.1 ms。
- **难度**：**中**（要保证重算的数值路径与 forward 完全一致，否则梯度不可复现）。
- **V100 适用**：✅（无硬件依赖）。
- **验收判据**：显存峰值下降 ≥ 1.2 GB；backward 时间增加 < 8 ms；梯度与"存 P"版本 `rel_err < 1e-5`。

### P5 —— 单趟 online softmax + 不物化 P（FA 式 attention 前向）

- **技术名**：tiling + running max/sum + rescale，`O` 在寄存器/SMEM 中累加后一次写回。
- **来源**：[FA2 论文](https://arxiv.org/abs/2307.08691)、[FA3 论文](https://arxiv.org/abs/2407.08608)（仅借鉴算法，不借鉴硬件路径）。
- **预期收益**：**5–15 ms**（当前 S=512）；**S≥2048 时变成必需**（否则 P 物化爆显存）。
- **难度**：**中–高**（fp32 下寄存器压力大：`BM×D` 的 Q tile + `BM` 的 m/l + `BM×BN` 的 S tile，`BM=64, BN=64, D=64` 时约 64×64×4B = 16 KB/块，可接受）。
- **V100 适用**：✅ **但只能拿 1.2–1.5×**，不是论文里的 2–4×（§2.3 判据）。
- **验收判据**：单层 attention fwd 时间下降 ≥ 20%；`dram__bytes` 下降 ≥ 50%；与物化实现 `max_abs_err < 1e-5`；**端到端**下降 ≥ 1%。

### P6 —— fp16 存储 + fp32 累加（唯一能用 Tensor Core 的路线）

- **技术名**：Q/K/V/激活存 fp16，GEMM 用 `mma.sync.m8n8k4`（sm_70 的 HMMA），累加器 fp32；配 loss scaling。
- **来源**：[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)（`.f16` + `.m8n8k4` 专为 sm_70 优化）；[FA README](https://github.com/Dao-AILab/flash-attention)（bf16 需 Ampere，反证 V100 只能 fp16）。
- **预期收益**：GEMM 理论峰值大幅提升（V100 的 fp16 TC 远高于 fp32 CUDA core）；**但**数值风险高、要重写全部 GEMM、要验证收敛。
- **难度**：**高**（2–4 周）。
- **V100 适用**：✅（fp16 可以；**bf16 不行**，见 §2.1）。
- **验收判据**：1000 步训练 loss 曲线与 fp32 基线相对偏差 < 0.5%；无 NaN/Inf；端到端 ≥ 1.5×。

### P7 —— Elementwise 链融合 + 向量化

- **技术名**：把 `residual add + RMSNorm` 合成一个 kernel；`RoPE` 与 `QKV` 的写回合并；所有 elementwise 用 `float4` 向量化 + `__ldg`。
- **来源**：[Liger 论文](https://arxiv.org/abs/2410.10989)（kernel fusion / in-place / chunking 三件套）、[Liger README](https://github.com/linkedin/Liger-Kernel)。
- **预期收益**：**15–50 ms**（若 H4 成立）。
- **难度**：**低–中**。
- **V100 适用**：✅。
- **验收判据**：单 kernel 有效带宽 ≥ 600 GB/s（用 `cudaMemcpy` 同尺寸对比）；kernel 总数下降 ≥ 30%；端到端 ≥ 2%。

### P8 —— Split-K / Stream-K（针对 dW 类 GEMM）

- **技术名**：K 维切分 + 两阶段 reduce；或 stream-K 调度。
- **来源**：[Stream-K, PPoPP'23](https://doi.org/10.1145/3572848.3577479)、[CUTLASS](https://github.com/NVIDIA/cutlass)。
- **预期收益**：**7–40 ms**（取决于有多少 shape 的 occupancy < 80%）。
- **难度**：**中**。
- **V100 适用**：✅。
- **验收判据**：`(896,896,2048)` 这类 shape 的 TFLOPS 从 X 提升到 ≥ 0.85 × 同卡大 shape 的 TFLOPS。

### P9 —— Fused linear CE（**仅显存保险**）

- **技术名**：chunked logits + 在线 log-sum-exp，不物化 `[2048,151936]`。
- **来源**：[Liger FLCE](https://github.com/linkedin/Liger-Kernel)、[Cut Cross-Entropy](https://arxiv.org/abs/2411.09009)。
- **预期收益**：**省 1.2–3.7 GB 显存；时间 +50 ms（净亏，见 §3.3）**。
- **难度**：**中**。
- **V100 适用**：✅（但**不推荐**在显存充足时启用，见 §3.3）。
- **验收判据**：峰值显存下降 ≥ 1 GB；loss 与基线 `rel_err < 1e-6`；**backward 时间不得增加 > 60 ms**（否则回滚）。

### 汇总表

| # | 技术 | 预期收益（ms） | 难度 | V100 适用 | 优先级理由 |
|---|---|---:|---|---|---|
| P0 | 分 shape GEMM 表 + nsys | 0（决策） | 低 | ✅ | 决定后续方向 |
| P1 | 显存池 + CUDA Graph | 30–150 | 中 | ✅ | 系统性开销，风险低 |
| P2 | Stride + GQA 索引 | 5–15 | 低 | ✅ | 零数值风险，必做 |
| P3 | causal 标志化 | 2–8 | 低 | ✅ | 顺手做 |
| P4 | backward recompute | −4（换 1.4 GB） | 中 | ✅ | 性价比最高 |
| P5 | online softmax 不物化 P | 5–15 | 中–高 | ✅（1.2–1.5×） | 为长 S 铺路 |
| P6 | fp16 + Tensor Core | 数百（理论） | 高 | ✅（fp16 only） | 唯一质变路线 |
| P7 | elementwise 融合/向量化 | 15–50 | 低–中 | ✅ | 便宜 |
| P8 | Split-K / Stream-K | 7–40 | 中 | ✅ | 视 P0 结果 |
| P9 | FLCE | +50（换显存） | 中 | ✅ | 仅 OOM 时 |

---

## 7. 验证计划（怎么证明有效，而不是"感觉快了"）

1. **微基准（固定 shape、`cudaEvent` 计时、100 次取中位数、warmup 20 次）**：单层 fwd / 单层 bwd / 单个 GEMM shape。
2. **端到端**：完整训练步（fwd+bwd+opt），跑 50 步取中位数，与 1.166 s 基线比。
3. **数值等价**：用 fp64 参考实现（NumPy 或 CPU 双精度）算单层，要求 `max_abs_err < 1e-5`；梯度用有限差分抽检 10 个元素。
4. **流量证据**：`ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum` 或 `nsys` 的 kernel 级流量估算，证明"优化真的减少了搬运"，而不是"换了计时方式"。
5. **回归**：`S ∈ {1, 7, 127, 512, 513, 1024}`、`B ∈ {1, 4}`、`HKV ∈ {2, 14}`、`is_causal ∈ {0,1}` 全交叉。

---

## 8. 存疑与未验证项（必须标注，避免把推测当事实）

1. **V100 的 shared memory 上限（96 KB/SM、单块最大 96 KB 需 opt-in）我没有拿到 NVIDIA 官方页面的直接引用**（`nvidia.com` 的 V100 产品页当前重定向到通用页）。**请用 `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)` 实测后再定 tile 大小**，不要照抄本文的 96 KB。
2. **V100 的官方 fp32/fp16 峰值 TFLOPS 同样未取到一手链接**。本文所有折算都只用你们**实测的 10.9 / 11.3 TFLOPS**，不依赖理论峰值。
3. **§1.3 的 H1–H5 是假设，不是结论**。它们由 FLOPs 与带宽算术推出"这些项加起来不够解释 0.49 s"，但**具体是哪一项，必须靠 P0 的 `nsys` 数据裁定**。
4. **Unsloth 的 2×/70% 不能归因于 kernel fusion**（含 4-bit、手动 autograd、checkpointing）。**Liger 的 20%/60% 是 vs HuggingFace 实现**，对自研引擎不可直接套用。
5. **FLCE 是否负收益，取决于你们是否已经物化 probs**。如果当前实现是"logits 和 probs 各存一份"，那 FLCE 省得更多（2.48 GB），但 backward 仍要多一次 GEMM；结论方向不变。
6. **FA2/FA3 的具体 backward 加速比在不同 head_dim / S 下差异很大**，本文只引用论文在 A100/H100 的报告值，**没有 V100 fp32 的公开数据**——这本身就是一个空白，也是你们可以做微基准发论文的点。

---

## 9. 参考链接（全部为本文实际抓取/核验过的页面）

**FlashAttention 系列**
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (arXiv 2307.08691)](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (arXiv 2407.08608)](https://arxiv.org/abs/2407.08608)
- [Dao-AILab/flash-attention（README：支持矩阵、FA4=CuTeDSL）](https://github.com/Dao-AILab/flash-attention)
- [PyTorch blog: FlexAttention + FlashAttention-4（2026-03-05）](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- [Flex Attention: A Programming Model for Generating Optimized Attention Kernels (arXiv 2412.05496)](https://arxiv.org/abs/2412.05496)

**训练侧融合**
- [Liger Kernel: Efficient Triton Kernels for LLM Training (arXiv 2410.10989)](https://arxiv.org/abs/2410.10989)
- [linkedin/Liger-Kernel（README：20%/60%、FLCE 基准表）](https://github.com/linkedin/Liger-Kernel)
- [Cut Your Losses in Large-Vocabulary Language Models (arXiv 2411.09009)](https://arxiv.org/abs/2411.09009)
- [unslothai/unsloth（README：2×/70%）](https://github.com/unslothai/unsloth)

**硬件能力判据（最重要的一手来源）**
- [PTX ISA（`cp.async` requires sm_80；`wgmma` requires sm_90a；`mma.m8n8k4.f16` optimized for sm_70；bf16/tf32 requires sm_80）](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUDA Graph 量化收益（launch overhead 消除、SM 利用率 78%→95%）](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/quantitative-benefits.html)
- [Volta Compatibility Guide（Independent Thread Scheduling 等 sm_70 行为差异）](https://docs.nvidia.com/cuda/volta-compatibility-guide/index.html)

**GQA / 视图语义 / 调度**
- [GQA: Training Generalized Multi-Query Transformer Models (arXiv 2305.13245)](https://arxiv.org/abs/2305.13245)
- [pytorch#154363：SDPA 的 GQA 广播是 memory footgun](https://github.com/pytorch/pytorch/issues/154363)
- [PyTorch SDPA 文档（`enable_gqa` / `is_causal` 的等价参考实现）](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Stream-K: Work-centric Parallel Decomposition for Dense GEMM (PPoPP'23)](https://doi.org/10.1145/3572848.3577479)
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)

**模型配置**
- [Qwen2.5-0.5B config.json](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json)

---

*本文所有"我们的估算"均为按 config 与带宽做的算术推导，已在文中标注公式；凡未实测的数字都标了"估算/上界"。**没有修改任何仓库文件**，本报告为唯一产物。*
