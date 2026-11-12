# 瓶颈定位：注意力（2026-11-12）

> 方法：把 B=8 / S=2048 / D=1280 / NH=10 / dh=128 / FF=5120 的一个 block 拆成阶段，
> fp32 与 `TG_FP16=1` 各跑独立进程（`tools/profile_block.py`，V100）。
> 目的：回答"fp16 只给 1.11×，torch 给 2.17×，差距在哪"。

## 1. 实测（每阶段取 5 次均值）

| 阶段 | fp32 (ms) | fp16 (ms) | 加速 |
| --- | --- | --- | --- |
| qkv_gemm | 11.79 | 2.50 | **4.7×** |
| gate_gemm | 15.94 | 2.86 | **5.6×** |
| up_gemm | 15.92 | 2.93 | **5.4×** |
| down_gemm | 15.57 | 3.56 | **4.4×** |
| out_gemm | 3.98 | 0.88 | **4.5×** |
| **attn_sdpa_only** | **30.53** | **30.46** | **1.00×** |
| attn_layout_split | 3.09 | 3.35 | 0.9× |
| attn_layout_back | 0.22 | 0.22 | 1.0× |
| silu_mul | 2.95 | 2.96 | 1.0× |
| **合计** | **99.99** | **49.71** | **2.01×** |

## 2. 三条结论

**① fp16 的 GEMM 已经打满，不是瓶颈。** 4.4–5.6× 已经接近 Tensor Core 的理论比。
想再压 GEMM 收益极小 —— **不要在这里花时间**。

**② 注意力是唯一的大头，而且 fp16 完全没帮上它。**
`attn_sdpa_only` 在两种模式下都约 30.5 ms；在 fp16 模式里它占整步 **61.3%**。
它没被优化，整步加速比就会被摊薄（实测 2.01×，其中 GEMM 贡献 5.1×）。

**③ 布局搬运不是问题。** 之前把 33 ms 全算作注意力是错的：
拆开后 `attn_layout_split` 只有 3.1 ms、`attn_layout_back` 0.22 ms。
**30.5 ms 是注意力数学本身。**

## 3. 为什么 PyTorch 快得多（读了源码，改过我最初的判断）

PyTorch 有四个 SDPA 后端（`torch/backends/cuda/__init__.py::sdp_kernel`：
`FLASH_ATTENTION` / `EFFICIENT_ATTENTION` / `CUDNN_ATTENTION` / `MATH`）。
我最初以为"torch 快是因为 flash + fp16"，**读了源码后发现对 V100 是错的**：

| 文件:行 | 原文 | 含义 |
| --- | --- | --- |
| `cuda/sdp_utils.cpp:535` | `check_sm_version<sm80, sm121>(dprops)`<br>`"Flash attention only supports gpu architectures in the range [sm80, sm121]"` | **flash 不支持 V100（sm70）** |
| `cuda/sdp_utils.cpp:1246` | `less_than_sm80_mem_efficient_dtypes = {kHalf, kFloat}` | 在 sm<80 上走 **mem_efficient**，且**fp32 也支持** |
| `cuda/sdp_utils.cpp:363` | mem_efficient 只要 `dh % alignment == 0`（fp16 时 8）、无 256 上限 | **dh=128 可以用** |
| `cuda/sdp_utils.cpp:431` | `if (dprops->major < 8) return true;` | sm70 上连数据对齐都不要求 |

⇒ **V100 上 torch 用的是 mem_efficient（xFormers 的 cutlass FMHA），fp32 也走它。**
所以它快的原因**不是 fp16/Tensor Core，而是"分块 + 融合 softmax、不物化 `[S,S]`"**。

**这条改变了我们的做法**：不必先做 fp16 注意力（那是 Ampere+ 才有收益的路），
**把 fp32 的分块融合注意力铺开到 dh=96/128 就能拿到大部分收益**，
而且顺带把 `[BH,S,S]` 的物化（B=8/S=2048/dh=128 时 335 MB×2）消掉 —— 那正好是我们"显存偏胖"的一块。

## 4. ORV 已有的基础与缺口

`python/tgai_go/train/kernels/tgai_attn_nop.cu`（47 KB）**已经实现了分块 + 在线 softmax**
（就是 flash 的思路，命名 `nop` = no-probability-store），但：

| 缺口 | 位置 |
| --- | --- |
| `#define DH 64` 写死 | `tgai_attn_nop.cu:19` |
| 只接受 fp32 | 内核里是 `float` / `float4` |
| 启动器硬性拒绝 | `cuda.py::_attn_flash_launch`：`if dh != 64: raise` |
| 只支持 causal、三维同形状 | `__init__.py::_sdpa_lowmem_ok` |

**而 smem 是硬约束**：V100 单 block 动态 smem 上限 **96 KB**（`_ATTN_FLASH_MAX_SMEM = 96*1024`）。
dh=64 时现有常量是 69632 B（68 KB）。若直接按比例放大到 dh=128，需要约 136 KB ⇒ **放不下**。

⇒ 必须**按 dh 选 tile**。粗算（fp32，一块 Q 块 + K/V 双缓冲）：

| dh | BQ | BK | smem 粗估 | 是否可行 |
| --- | --- | --- | --- | --- |
| 64 | 64 | 64 | 68 KB（现状） | ✅ |
| 96 | 64 | 64 | ≈ 100 KB | ❌ 超；需 BK=32 |
| **128** | 64 | 32 | ≈ 80 KB | ✅ |
| 128 | 32 | 64 | ≈ 96 KB | ⚠️ 顶格 |

所以 dh=128 要用 **BK=32**（或 BQ 减半）换空间。这也说明：**若先上 fp16 存储，smem 减半，
dh=128 就能用更舒服的 tile** —— fp16 的价值在这里是"换 tile 空间"，不只是速度。

## 5. 计划（按可验证性拆，顺序已按源码结论调整）

| 步 | 内容 | 判据 |
| --- | --- | --- |
| **1** | 把 `tgai_attn_nop.cu` 的 `DH` 模板化，**先只做 dh=64 的回归** | 与现有实现逐位一致（同一份输入） |
| **2** | **dh=128 + fp32，BK=32**（先不碰 dtype） | 与物化路径相对误差 ≤1e-3；`attn_sdpa_only` 明显下降 |
| **3** | dh=96（覆盖常见配置） | 同上 |
| **4** | 可选：fp16 存储（smem 减半，可放宽 tile） | 误差可报告，且不掉速 |
| **5** | 接进 `scaled_dot_product_attention` 自动选择，保留物化路径兜底 | 不改用户代码即生效 |

**为什么把 fp32 排在 fp16 前面**：V100 上 torch 的 mem_efficient **fp32 也支持**，
所以"不物化 + 融合 softmax"本身就是主要收益；fp16 在那个基础上只多一层。

**预期收益**：若注意力从 30.5 ms 降到 torch 量级（个位数 ms），
整步 **49.7 → ~23 ms（约 2.1×）**，即在 5,667 tok/s 的基准上反超；
同时省掉 335 MB×2 的物化，显存也跟着降。

**风险与纪律**：
* 内核正确性优先 —— **先用小形状（S=64/128、NH=2、DH=64/128）与物化路径逐元素对拍**，
  再放大到 S=2048；
* causal 的边界块必须单独验（`q_start > k_end` 的块要整块跳过，不能靠 softmax 抵消）；
* fp16 累加必须在 fp32 里做，否则误差会掩盖成"数值没问题"。

## 6. 复现命令

```bash
python tools/profile_block.py            # fp32
python tools/profile_block.py --fp16     # 开 TG_FP16=1（必须独立进程）
```
