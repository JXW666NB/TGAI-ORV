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
