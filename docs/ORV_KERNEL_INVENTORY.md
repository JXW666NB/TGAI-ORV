# ORV 引擎既有能力清单（写代码前先查这里）

> **这份文档的唯一目的：避免重复造轮子。**
> 2026-11-14 一天之内，我（agent）在同一个引擎里重复造了四次已有的东西：
> 融合 causal softmax、原生 flash 注意力、fp16/bf16 动量 Adam、分组 GEMM。
> 每次都是"先写新的，后来发现旧的已经在仓库里"。
>
> **规矩：动内核之前，先在本文件里搜关键词，再 `grep` 源码确认。**
>
> 自动生成原始清单：`python tools/inventory_orv.py`（只读，扫 `cuda.py` 与模块头）。

---

## 0. 一句话现状

`cuda.py` **7,653 行**，含 **114 个 CUDA 内核**、**245 个启动器**；`train/` 下 **68 个模块**。
绝大多数"我需要一个 X 内核"的场景，X 已经存在。

---

## 1. 内核按前缀分组（114 个）

| 前缀 | 数量 | 覆盖能力 |
| --- | --- | --- |
| `ewise_` | 28 | 逐元素：加/乘/缩放/exp/log/tanh/relu…（含 vec4 分支） |
| `moe_` | 14 | **MoE 全套**：topk / sigmoid / gather_pairs / scatter_pairs / compact / gather_rows / scatter_rows / gather_slots / scatter_slots / row_scale / row_dot / pad_rows |
| `emb_` | 9 | 嵌入前向与**多路反向**（hist / plan / fill / owned，含确定性分支） |
| `fused_` | 6 | 融合 RoPE（标量 + vec4）、融合原地残差加 |
| `silu_` | 4 | SwiGLU 前向/反向（含原地版） |
| `cast_` | 4 | fp32↔fp16、fp32↔bf16（RTNE，与主机侧逐位一致） |
| `adamw_` | 4 | AdamW：单张量 / 多张量融合 / **fp16 动量** / **bf16 动量** |
| `ce_` | 3 | 交叉熵（按行 / 取均值 / 反向） |
| `sumsq_` | 3 | 三段式平方和（顺序 / 分块 / 汇总） |
| `rmsnorm_` `rms_` | 5 | RMSNorm 前反向、rstd、dW 确定性归约 |
| `attn_` | 2 | **因果掩码 + softmax 融合**（已改成一块一行 + warp 归约） |
| `softmax_` `log_softmax_` | 4 | 通用 last-dim softmax / log-softmax 及其反向 |
| `rope_` | 2 | 半旋转式 RoPE（原地 + 反向） |
| `rowwise_` | 2 | **行内 max/argmax**（2026-11-14 新增，替代 D2H + one-hot） |
| `situ_` | 2 | SiTU-GLU 融合激活（前向 + 反向） |
| `index_` `gather_` `scatter_` | 8 | 索引查表 / 行收集 / scatter-add（含 8 路索引） |
| `cat_` `permute8_` `transpose2d` `swap_last2` `broadcast_to8` | 10 | 布局搬运 |
| `sgemm_` | 2 | cuBLAS 组合（`sgemm_mode` / `sgemm_compose`） |
| `muon_` | 1 | Muon 动量 |
| `sum_axis0_rows` `sum_axis8` `reduce_sum` `repeat_reduce` | 4 | 归约 |
| `bcast_add4` | 2 | 广播加 |
| `gemm_f16_*_raw` | 3 | **fp16 输入 + fp32 累加**的 NT/NN/TN 三条 GEMM（零转换） |

---

## 2. 我重复造过的四个轮子（附正确入口）

| 我以为没有 | 其实在 | 正确入口 |
| --- | --- | --- |
| 融合的因果 softmax | `attn_mask_softmax` / `attn_mask_softmax_bwd` | `T.attn_mask_softmax(scores, causal=True)`（**模块级**，不在 `T.functional` 里） |
| 原生 flash 注意力内核 | `training/kernels/tgai_attn_nop.cu`（`attn_fwd_stats` / `attn_bwd_flash`） | `attention_orv(..., lowmem=True)`（**要求 d_k=64**） |
| fp16 动量的 Adam | `adamw_update_multi_f16` + `opt_fp16mv.Fp16MomAdamW` | `Fp16MomAdamW(params, ...)` |
| bf16 动量的 Adam | `adamw_update_multi_bf16` + `Fp16MomAdamW(state="bf16")` | 同上，`state="bf16"`（**fp16 会下溢，用这个**） |
| 分组 GEMM 做 MoE | `_grouped_sgemm` + 14 个 `moe_*` 内核 | `rt.grouped_nt(...)`（**当前 rc=7，不可用**，见 §4） |

**还有两个我根本不知道存在的模块**：
- `int8_adamw.py` —— int8 分块 AdamW，**省优化器状态显存**的现成方案（目标②的另一条路）
- `amp_mlp.py` / `amp_api.py` —— 真 AMP（激活存 fp16）与 `T.amp()` 上下文

---

## 3. Python 层现成能力（别再自己写）

| 需求 | 用这个 |
| --- | --- |
| 归约 `max/min/var/std/norm` | `redux_extra.py`（`Tensor.max(dim)` 等，CUDA 上已走内核） |
| 优化器 | `T.optim.AdamW / LAMB / Muon`（`T.optim` 只有这三个内置名）；`opt_extra` 提供 `SGD / Lion / Adafactor / Accumulator`（需显式 import）；`opt_fp16mv.Fp16MomAdamW`；`int8_adamw` |
| 学习率调度 | `sched.py` |
| 检查点 | `ckpt.py`（权重 + 优化器状态 + 轮转）、`checkpoints.py`（GGUF/safetensors/npz） |
| CUDA Graph | `graph_train.GraphStep`（整步）、`layer_graph.py`（逐层反向） |
| 分布式 | `dist.py`（广播 + 全归约）、`dist_p2p.py`（GPU P2P） |
| 数据 | `data.py`、`loader.py`、`sft_data.py` |
| 算子插件 | `ops_plugin.py` / `ops_bundle.py` / `ops_cuda.py`（NVRTC 编译用户 .cu） |
| 自检 | `doctor.py` |
| 与 torch 互转 | `interop.py` |
| 分块注意力（更早的实现） | `v3_attn_chunked.py`、`v3_flash.py`、`v3_flash_dev.py` |

---

## 4. 已知的坑（全部实测，别再踩）

| 坑 | 事实 | 结论 |
| --- | --- | --- |
| 原生 flash 内核 | `#define DH 64` 编译期写死；`d_k=128` 时**静默不输出**；`d_k=64` 时可用但**整模型 1.67 s/层**（物化只要 0.09 s） | **不要**切默认路径；单算子 0.015 s 是假象（内部回退） |
| fp16 动量 | fp16 最小正规数 6.1e-5，二阶动量常为 1e-8 ⇒ **下溢为 0**，6 步 loss 8.42→**552.98** | 用 **bf16** 动量（同 2 字节，同指数范围） |
| 分组 GEMM | `cublasSgemmGroupedBatched` 两种签名（15 参数 / 16 参数含 groupSize）**都返回 rc=7**；句柄有效；运行时**没有** Ex 版符号 | 判定不可用，**不要再试签名** |
| `max` 归约 | 旧实现 = D2H + 主机 argmax + one-hot H2D，每对块 2 次强制同步 | 已换 `rowwise_max` 内核（2026-11-14） |
| fp16 GEMM | `TG_FP16=1` 数值正确但**不提速**（0.50 vs 0.49 s） | 在这张 V100 上无收益 |
| CUDA Graph 收益 | 每步约 2,400 内核 + 1,200 cuBLAS ≈ 36 ms（占 8%） | 上限个位数百分比，**不是** 1.71× |
| 检查点边界 | `to_numpy()` 往返每层一次强制同步 | 用 `Tensor._from_gpu(buf)` 包装同块显存成新叶子，零拷贝 |
| 掩码上传 | 每对块重建并上传 8 MB | 按 `(rows, cols, offset)` 缓存，S=2048 只需 7 张 |

---

## 5. 真正**不存在**的东西（可以放心造）

以下是我确认过的空白，造之前也请再 grep 一次：

1. **块内融合的在线 softmax 内核**（mask+exp+rescale 一趟完成，替掉当前 6 个 pass）
2. **d_k=128 可用的 flash 内核**（现成的写死 64；DH=128 需把 BT 从 64 降到 32 才放得进 96 KB smem）
3. **MoE 的 batched/grouped 分派**（14 个 `moe_*` 是配套件，但把它们串成"一次 GEMM 覆盖全部专家"的上层还没写）
4. **fp16/bf16 权重主副本 + fp32 主权重混合训练**（`amp_mlp` 只覆盖 MLP 块）

---

## 6. 维护约定

- 每新增一个内核/模块，**同步更新本文件的 §1 或 §3**（一行即可）。
- 每次因"不知道已有"而重复造轮子，**在 §2 或 §4 加一行**，写清"我以为没有 → 其实在 → 正确入口"。
- 本文件与 `tools/inventory_orv.py` 的自动输出互为校验：数字对不上就重新生成。
