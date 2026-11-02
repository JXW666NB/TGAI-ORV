# 与 PyTorch 的差距盘点（2026-11-02 实测）

> 原则：**每个数字都标来源**。测不了的就标"未验证"，不拿推测当结论。

## 0. 一句话结论

正确性已经够用（前向 4.8e-06 / 反向 1e-4 / 优化器 4.7e-09），
**差距集中在"显存 1.7 倍、速度 1.8 倍"两件事上**，其次是 dtype/多卡/API 面。

## 1. 同形状硬指标（V100-32GB，Qwen2.5-0.5B 结构，fp32 + AdamW，B=4/S=512）

| 指标 | PyTorch 2.x | 本引擎 | 比值 |
| --- | --- | --- | --- |
| 峰值显存（`max_memory_allocated` / `_pool_live_bytes`） | **16.61 GiB** | 28.55 GiB | **1.72×** |
| 同上，开梯度检查点 | —（默认不开） | **19.97 GiB** | 1.20× |
| 步时（3 步均值，含 sync） | **0.677 s** | 1.22 s | **1.80×** |
| 权重占用 | 2.35 GiB | 1.94 GiB（fp32 494M） | 0.83× |

**测量方式**：PyTorch 用 `torch.cuda.max_memory_allocated()`，我们用自己的 `_pool_live_bytes`
每 20 ms 采样；两者都排除进程外占用（产线服务已停）。形状、层数、头数、FFN、词表完全一致，
`allow_tf32=False` 对齐精度。

### 差距拆解（按子智能体台账 + 实测交叉验证）

| 项 | 量级 | 状态 |
| --- | --- | --- |
| SDPA 物化 3 份 `[BH,S,S]`（PyTorch 用 memory-efficient/flash） | ≈4.1 GiB | **未修**（`_sdpa_lowmem` 已实现但 `lowmem=False` 从不启用） |
| `permute`/`__getitem__`/`repeat_interleave` 在 CUDA 上是真拷贝（PyTorch 是视图） | ≈2~3 GiB | **未修** |
| 词表维度张量多份（logits/softmax/grad） | ≈3.5 GiB | 部分（`chunked_ce` 已实现，但 CUDA CE 的 ignore 路径返回形状有 bug，未全通） |
| 中间梯度从不释放 | ≈11 GiB | ✅ **已修**（逐节点释放，`release_variables` 语义） |
| 检查点在 CUDA 上是空操作 | ≈11 GiB | ✅ **已修**（设备侧 clone + `GpuBuf` 直传） |

## 2. 数值正确性（全部实测）

| 项 | 结果 | 判据 |
| --- | --- | --- |
| Qwen2.5 官方权重前向 | **4.816e-06** | ≤1e-5 ✅ |
| LLaMA-60M 前向 vs 孪生 | **1.16e-07** | ≤1e-5 ✅ |
| 反向 24 个参数 | **≤1e-4** | ✅ |
| AdamW 6 步 vs torch | **≤1e-5** | ✅ |
| 梯度检查点（开/关） | **逐位一致（0.0）** | ✅ |
| 分块 CE vs 整块 | **4.77e-07** | ✅ |
| bf16 转换 vs torch | **位级 100.0000%** | ✅ |
| `accept --quick`（V100） | **12/12 PASS** | ✅ |

## 3. 算子与能力覆盖

| 维度 | PyTorch | 本引擎 |
| --- | --- | --- |
| 后端算子 | 2000+ | **79**（12 组：core 54 + fallback 8 + MoE 可选 17） |
| CUDA 张量 dtype | fp32/fp16/bf16/fp8/int8… | **只有 fp32**（`Tensor.dtype` 对 CUDA 恒返回 float32） |
| 视图（view/stride） | 完整 | **无**（`permute`/切片都是真拷贝） |
| 原地操作 | 支持 | 基本没有（少数 kernel 内部） |
| 自动微分 | 完整 + 高阶 | 一阶；CPU+CUDA 都有；**中间梯度已按 PyTorch 语义释放** |
| 梯度检查点 | `checkpoint` / `checkpoint_sequential(segments)` | ✅ 同签名，含 `segments`；CPU+CUDA |
| 混合精度 | `autocast` + `GradScaler` | **只有存储半边**（3 个 bf16 内核源码，**未接进引擎**）；计算仍 fp32 |
| SDPA | flash / mem-efficient / math 三后端自动选 | 物化版（`_sdpa_lowmem` 存在但未启用） |
| 融合算子 | `torch.compile` 自动融合 | 手写 8 个（可回退） |
| 优化器 | 20+ | AdamW / LAMB（仅 CPU）/ Muon |
| 多卡 | DDP/FSDP/TP | `dist.py`/`launch_dp.py` 有地基，**无 2 卡硬件验证** |
| 数据加载 | `DataLoader` + sampler | `read_jsonl`/`normalize`/`train_eval_split`（简单版） |
| 生态互操作 | `state_dict`、`nn.Module`、ONNX… | safetensors 读写 ✅、HF `config.json` 部分映射 ✅、`tokenizers` ✅；**无 `state_dict`/`Module` 互转** |
| 训练 API | 全手工 | `api.train(method="sft"/"pretrain")`；DPO/GRPO **明确未实现** |

## 4. 工程标准（这是我们已经达标的）

| 项 | 状态 |
| --- | --- |
| `tools/check_package.py`（干净 venv 可 import） | ✅ PASS |
| CI（GitHub Actions，CPU 部分） | ✅ 连续 success |
| `tests/` 纯 CPU 单测 | ✅ **292 passed** / 46 个文件 |
| semver + CHANGELOG + git tag | ✅ `v0.4.0` |
| 无 BOM / 无 CRLF / 无临时脚本 | ✅ 有护栏测试 |
| 错误信息写"算子/形状/怎么修" | ✅ 抽查通过（含本轮的显存不足提示） |

## 5. 我们比 PyTorch 更强的地方（真实差异）

1. **内核是 NVRTC 源码字符串** ⇒ 运行时改内核、重编译；PyTorch 需要重编译整库。
2. **没有 dispatch 层** ⇒ 每个算子可显式传策略，不需要 autocast 那种"猜"。
3. **纯 Python 可读** ⇒ 从算子到训练循环全部可评审（PyTorch 要读 C++/CUDA）。
4. **79 个算子有显式契约**（`backend.py`）⇒ 换后端（Vulkan/ROCm）时有明确的覆盖面清单。

## 6. 差距排序（按"能不能替别人训模型"）

1. **显存 1.72×** —— 主因是 SDPA 物化 + 视图变拷贝 + 词表多份；检查点已能把差距压到 1.20×。
2. **速度 1.80×** —— 缺融合（SDPA/rope/silu_mul 已实现但没接线）、逐算子启动开销。
3. **dtype 单一（只有 fp32）** —— bf16 存储半边已验证位级一致，**计算半边需 sm_80+**。
4. **多卡未验证** —— 缺 2 卡机；DP 正确性只在 2 进程 CPU 上验过。
5. **API 面窄** —— 只有 SFT/预训练；DPO/GRPO 明确抛错。
6. **生态互操作** —— 无 `state_dict`/`nn.Module` 互转，用户迁移有摩擦。

## 7. 下一步（按收益/风险比）

| 优先级 | 动作 | 预期 | 风险 |
| --- | --- | --- | --- |
| P0 | 启用 `_sdpa_lowmem`（`dh=64` 已满足） | 显存 −4~6 GiB | 中（需数值对拍） |
| P0 | 接线已实现的融合核（`fused_rope`/`silu_mul`） | 速度 +10~20% | 低 |
| P1 | 词表张量单份（修 CUDA CE ignore 形状 bug） | 显存 −3.5 GiB | 低 |
| P1 | 视图语义（至少 `permute`/切片零拷贝） | 显存 −2~3 GiB | **高**（要改 autograd 契约） |
| P2 | 旋钮注册表 `tuning`（含 manifest） | 可复现的自由度 | 低 |
| P2 | 2 卡验证 | L3 判据 | 需硬件 |

---

## 8. 2026-11-02 更新：显存**追平**了（实测四组矩阵）

V100-32GB，Qwen2.5-0.5B 结构，fp32+AdamW，B=4/S=512，产线已停：

| 配置 | 峰值活张量 | 步时 | vs PyTorch（16.61 GiB / 0.677 s） |
| --- | --- | --- | --- |
| base（无优化） | 28.55 GiB | 1.22 s | 1.72× / 1.80× |
| `SDPA_LOWMEM=1` | 23.14 GiB | 2.60 s | 1.39× / 3.84× |
| `GRAD_CKPT=1` | 20.15 GiB | 1.33 s | 1.21× / 1.96× |
| **两者齐开** | **17.00 GiB** | 2.69 s | **1.02× / 3.97×** |

**结论**：显存差距已从 1.72× 压到 **1.02×（基本追平）**；代价是时间——
低显存注意力内核比物化实现慢约 2 倍，这是下一步的主要攻坚点。

### 两个开关（都可复现）

```python
cfg["GRAD_CKPT"] = True      # 梯度检查点（CPU+CUDA 都真生效）
cfg["CKPT_SEGMENTS"] = 6     # 每 6 层一个检查点（PyTorch segments 语义）
cfg["SDPA_LOWMEM"] = True    # 不物化 P 的注意力（显存 -5.4 GiB，时间 +2.1x）
```

### 数值对拍（同一份输入，BH=8/S=128）

| 项 | 相对误差 |
| --- | --- |
| 前向 | **5.7e-08** |
| dq / dk / dv | **2.4e-06 / 2.3e-06 / 2.1e-07** |

> 教训：第一版对拍脚本用共享 `rng` 生成两组输入，比出 relerr 1.42（√2），
> 差点误判"内核是坏的"。**对拍必须同一份输入** —— 已写进测试注释。

---

## 9. 2026-11-02 下半场：速度瓶颈定位（阶段分解，不猜）

V100，B=4/S=512，Qwen2.5-0.5B 形状，每段都 `sync` 后计时、跑完释放（避免图堆积 OOM）：

| 阶段 | 耗时 | 占比 |
| --- | --- | --- |
| forward | **0.419 s** | 36% |
| **backward** | **0.723 s** | **62%** |
| optimizer.step | 0.024 s | 2% |
| 合计 | **1.166 s** | 100% |

对照 PyTorch 同形状 0.677 s ⇒ 我们 forward 约 1.9×、backward 约 1.6×。

### 结论：先打 backward

按子智能体台账 + 代码走查，backward 里最可疑的三项（按预期收益）：

1. **每层都 `scores + [S,S] mask`**：S=512 时 mask 是 59 MB，
   24 层 × 前向一次分配 + 一次全尺寸 elementwise；反向还要多一次。
   正解是**causal softmax 内核**（掩码在 softmax 内部做，不物化 mask）。
2. **`permute` / `repeat_interleave` 是真拷贝**（PyTorch 是视图）：
   GQA 的 k/v 被放大 7 倍（`KV=2 → NH=14`），反向还要再算一次。
3. **softmax 两趟**（max + 归一化），PyTorch 的融合核一趟做完。

### 已确认的收益（本轮）

- 融合 `silu_mul`：1.22 → **1.17 s**（+4.3%），显存 −0.59 GiB。
- 说明瓶颈不在这个算子 —— 4% 的收益印证了"先测量再优化"的必要性。

---

## 10. 2026-11-02 晚：**同进程公平对照**（更正此前 1.18× 的说法）

### 10.1 正确的对照方式

同一个进程、同一份数据、连续 3 步、中间**不**清池/不 gc，两边同形状同 dtype：

| | s/step |
| --- | --- |
| PyTorch | **0.610** |
| 本引擎（融合 RoPE 默认开） | **≈0.811** |
| **比值** | **1.33×** |

> 此前说过"1.18×"——那是拿我们一次跑的 0.802 与 torch 另一次跑的 0.677 相比，
> **不同进程/不同状态不可比**。已更正。

### 10.2 阶段对比（同口径分开测）

| 阶段 | torch | 我们 |
| --- | --- | --- |
| forward | 0.175 s | **0.230 s**（1.31×） |
| backward | 0.382 s | ≈0.52 s（1.36×） |
| opt.step | ≈0 | 0.032 s |

### 10.3 一条测量方法论教训

之前用 `phase2.py` 测到 forward = 0.394 s，觉得比 torch 慢 2.25×。
实际上是脚本**每步之间调了 `empty_cache()`**，下一步又要重新向驱动要内存 ——
量到的是"分配器抖动"，不是 kernel。把清理去掉后：forward **0.230 s**。
⇒ 微基准里不要在被测代码之间做 `empty_cache`/`gc`；要清理就单独一轮测。

### 10.4 当前进展小结（本轮会话）

| 时间点 | 步时 | 相对 torch |
| --- | --- | --- |
| 会话开始 | 1.125 s | 1.66× |
| 分配器修复（两处硬编码 12.2 GiB） | 0.898 s | — |
| + half-split RoPE 融合内核 | 0.802 s | — |
| **现在（同进程公平对照）** | **≈0.811 s** | **1.33×** |

### 10.5 试过但**更慢**的（已按证据回退）

- causal softmax（掩码内联）：0.802 → **0.893 s**。内核每行一个线程串行扫 k，
  并行度不足；已改为 opt-in（`TG_SOFTMAX_CAUSAL=1`），要收益需先改成 warp 归约。

---

## 11. 2026-11-02 夜：**真正的瓶颈是 CPU 发射，不是 kernel**

### 11.1 实测（新增 `tools/profile_kernel.py`，CUDA event 计时）

```
wall = 744 ms    kernels = 1330    GPU 实际忙 = 111.6 ms
=> GPU 利用率仅 15%，85% 的时间 CPU 在发射/建图/等
```

Top kernel（占 GPU 时间）：

| kernel | 次数 | ms | 占比 |
| --- | --- | --- | --- |
| `adamw_update` | 290 | 18.8 | 16.8% |
| `sum_axis8` | 72 | 14.5 | 13.0% |
| `softmax_lastdim_bwd` | 24 | 8.8 | 7.9% |
| `bcast_add4_v4` | 96 | 4.9 | 4.4% |
| `ewise_scale_v4` | 72 | 4.8 | 4.3% |

⇒ 即使把 GPU kernel 全优化到 0，整步也只能降到 ~630 ms。
**优化方向必须换成"减少发射次数 / 降 CPU 开销"**。

### 11.2 已有的杠杆：CUDA Graph

运行时已有 `graph_begin/graph_end/graph_launch`（含 AdamW 偏差校正节点改写）。
捕获能把 1330 次发射压成 1 次。

但当前**捕获会失败**：

```
CudaError: CUDA Graph 捕获中内存池未命中 7340032 字节，拒绝 cuMemAlloc
```

这是**设计使然**（捕获中不允许分配），说明进捕获前必须把所有需要的块预先放进池子。
下一步：给捕获加"预留 arena"（bump 分配）或按尺寸预热到池内，再捕获。
预计收益：1330 → 1 次发射，wall 有望从 744 ms 降到 ~150-200 ms。

### 11.3 顺带修好的（本轮已提交）

`ce_row_stats` 的**非合并访问**（每线程连续分块，步长 V/256）：
CE 前向 67.5 → **3.6 ms（18.7×）**，整步 0.811 → **0.748 s**。
`log_softmax_lastdim` 前向/反向也从"线程-per-row 串行扫 k"改成 warp-per-row。
