# 记事本 · 多厂商训练与「十行代码」路线（想法存档）

> ⚠️ **2026-11-12 更正（重要，先读这条）**
>
> 本文 §3 把 **TG Link** 定位成「host-staged 集合通信层」。**那是退路，不是原始设想。**
>
> **TG Link 的原始设想是：用 Vulkan 让显卡与显卡之间 P2P，实现多厂商交火。**
> 也就是说，目标是不经主机中转的**真 P2P**；host-staged all-reduce 是这条路走不通时的兜底。
>
> 两者差别很大：真 P2P 能解锁**跨厂商 TP（张量并行）**，而 host-staged 只够做 DP/PP。
> 是否可行要靠实验判定，判据见 `docs/INDEX.md` 九点六 §A。
>
> 另外注意：**「后端混用」（一个进程里用多种后端）是另一件事**，本文不覆盖它。

> 2026-11-01 记录。**当前不执行**：先把 CUDA 这条线做实（见 §6）。
> 这份文件的用途是：把方向、物理约束、已有资产、里程碑和判据先钉住，
> 避免以后重复论证或高估可行性。

## 1. 目标（一句话）

让"没有资源、没有框架"的人也能训练/微调模型：
**10 行以内代码**完成 预训练 / SFT / DPO / GRPO，并且**能跑在手头任何显卡上**（N/A/Intel 混插）。

## 2. 三层目标与它们的真实难度

| 层 | 内容 | 难度 | 说明 |
| --- | --- | --- | --- |
| API 层 | `tgai_go.api.train(model, data, method="sft"/"dpo"/"grpo", ...)` | **中** | 我们已有 SFT 全链路；DPO 需 reference model + 偏好对；GRPO 需 rollout 采样 + reward，工作量最大 |
| 后端层 | Vulkan 计算后端（训练，不只是推理） | **大** | 生态空白 ⇒ 既没现成轮子也没人替你踩坑；但正因为空白才有价值 |
| 硬件层 | 混杂厂商多卡（「TG Link」） | **大但可行**（见 §3） | 关键结论：**DP 不需要 GPU 间直连** |

## 3. TG Link 的物理现实（先把话说清楚）

**不能做的事**：跨厂商 GPU 之间的"直连"。
- NVLink 只有 NVIDIA；AMD 的 Infinity Fabric 只有 AMD；Intel 同理；
- PCIe P2P（GPU 直接读对方显存）跨厂商通常被驱动/IOMMU 禁止，Windows 上更严格；
- 所以"绕过 ROM/驱动让不同厂商显卡直接互传"**在通用意义上做不到**。

**但关键洞察是：数据并行（DP）根本不需要直连。**

| 量 | 数值 | 结论 |
| --- | --- | --- |
| 0.5B 模型 fp32 梯度 | ≈ 1 GB/步（每参数 4 B） | — |
| PCIe 3.0 x16 实测有效带宽 | ≈ 10–12 GB/s | 单步全归约 ≈ 0.1 s |
| 我们的步时（V100，batch2/seq256） | 0.8 s/步 | **开销 ≈ 12%** |

⇒ **把梯度经 host 内存中转做 all-reduce，混杂厂商的 DP 是现实可做的**，
不需要任何"厂商私有直连"。真正不现实的是 **TP（张量并行）**（要低延迟高带宽）；
**PP（流水线并行）**只传激活，量小，host 中转也可行。

**因此 TG Link 的正确定位**：
> 厂商中立的 **host-staged 集合通信层**：pinned memory + 异步拷贝 + 分层 all-reduce
> （先单机多卡，再跨机）；Linux 上后续可用 `dma-buf` / `VK_KHR_external_memory` 做零拷贝优化。

## 4. 我们已有的资产（不是从零开始）

| 资产 | 位置 | 现状 |
| --- | --- | --- |
| 后端抽象（79 算子 / 12 组） | `python/tgai_go/train/backend.py` | ✅ 已有，`required_methods()` 79 条 |
| 参考后端模板 | `backends/numpy_ref.py`（36/79） | ✅ 可当语义参照 |
| Vulkan **推理**内核 | `shaders/*.comp` + 12 个 `.spv` | ✅ 已能跑（add/attention/mm_bf16*/rmsnorm/rope/silu_mul/softmax/moe_router） |
| SPIR-V 运行时加载 | `src/tgai_vk_kernels.c`（`load_spv(dir,name)`） | ✅ C 侧已实现 |
| 数据并行骨架（共享内存 + 文件屏障） | `train/dist.py` | ✅ 正确性已用 2 进程 CPU 验证 |
| 训练链路（SFT/续训/检查点） | `tgai_go.train` + `examples/qwen25_sft.py` | ✅ 已对齐到 1e-5（前向）/1e-4（梯度） |

**最关键的缺口**：
1. **训练侧 Vulkan 内核 = 0**（所有反向都要自己写：rmsnorm_bwd、softmax_bwd、GEMM 的 dX/dW、
   embedding scatter-add、CE 反向…）；
2. **本机没有 SPIR-V 编译器**（`glslc`/`glslangValidator` 都没有），
   仓库里有 `tools/build_glslang.bat`（从源码构建 glslangValidator，需 VS2022 + CMake + Ninja）；
3. 训练后端要在 Python 侧把 Vulkan 用起来（现在 Python 只驱动 CUDA 的 ctypes 路径）。

## 5. 里程碑（每个都有可测判据，不许"差不多了"）

| 阶段 | 内容 | 判据 |
| --- | --- | --- |
| **V0** | 装 glslang → 写一个 fp32 GEMM 计算着色器 → Python(ctypes) 驱动 Vulkan 跑通一次矩阵乘 | 与 numpy 对拍 rel ≤1e-5；本机核显（AMD Radeon, Vulkan 1.3.260）上跑通 |
| **V1** | 补反向：GEMM 的 dX/dW + rmsnorm_bwd + softmax_bwd | tiny MLP 训练收敛，参数与 CPU 版差 ≤1e-4（10 步） |
| **V2** | `backends/vulkan.py` 接入 `backend.py`（先覆盖训练必需子集） | `TG_BACKEND=vulkan python tools/accept.py --quick` 通过 |
| **V3** | **TG Link v0**：host-staged all-reduce，两进程（CUDA + Vulkan 各一）跑 DP | 2 卡加速比 ≥1.3×（混杂卡先求能跑，再求快）；梯度与单卡一致 |
| **V4** | 10 行 API：`train(...)` 封装 pretrain/SFT | 文档里的 10 行示例可复制运行 |
| **V5** | DPO / GRPO | 同一份数据上 DPO 能降 loss；GRPO 能跑完一轮 rollout+update |

## 6. 为什么现在不做（顺序问题，不是技术否决）

- **CUDA 这条线还没收完**：真 AMP（bf16 激活存储）、2 卡加速比、以及 `.TG` 加载正确性；
- Vulkan 训练是"从 0 造轮子"，一旦开始会长期占用注意力，
  而 CUDA 侧每收一项都直接提升"能替别人训练"的可信度；
- 先把 CUDA 做成"可交付"，Vulkan 才有一个**能对拍的正确性基准**（这是它的最大价值来源）。

## 7. 一句话立场

> 技术平权不是靠喊"人人可训"，而是靠**把门槛一层层拆掉并且每层都能验证**：
> 先 CUDA 对齐到 1e-5、再 Vulkan 单卡、再多厂商 DP、最后才是 10 行 API。
> 顺序错了就会得到一堆跑不起来的 demo。
