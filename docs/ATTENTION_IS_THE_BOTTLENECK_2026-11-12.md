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

⇒ **V100 上 torch 用的是 mem_efficient（xFormers 的 cutlass FMHA）**。
但真正决定"快不快"的是它的 **tile 实例化表**（`mem_eff_attention/kernels/cutlassF*.h`），
模板参数是 `AttentionKernel<元素, 架构, 对齐, kQueriesPerBlock, kKeysPerBlock, kMaxK, ...>`，
其中 **`kMaxK` 就是该变体支持的 head_dim 上限**：

| dtype | 架构 | 实例化（Q×K, kMaxK） |
| --- | --- | --- |
| **fp32** | Sm50 / **Sm70** / Sm75 | **只有 `64×64, kMaxK=64`** |
| fp16 | Sm70 | `64×64, kMaxK=64`｜**`32×128, kMaxK=128`**｜`32×128, kMaxK=65536`（gmem 流式）|
| bf16 | Sm80 | `64×64, kMaxK=64`｜`64×128, kMaxK=128`｜`32×128, kMaxK=65536` |

**所以对 dh=128 / V100 这个具体情况**：

| 路径 | 有没有快路 |
| --- | --- |
| fp32 | ❌ `kMaxK=64` ⇒ **torch 自己也退回物化（MATH）** |
| fp16 | ✅ `32×128, kMaxK=128` |

**这条推翻了我上一版的修正**：我先前说"fp32 也支持所以 fp32 就该够"，错在只看了 dtype 列表、
没看 tile 表的 `kMaxK`。**结论回到：dh=128 要有快路，必须让内核吃 fp16。**

而 dh=128 的 tile 几何也有了权威依据：**`BQ=32, BK=128`** ——
正好等于我从 smem 预算推出来的"查询块减半"。这次不是推测。

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

## 5. 计划（按源码结论排，**顺序已改两次，这是最终版**）

| 步 | 内容 | 判据 | 状态 |
| --- | --- | --- | --- |
| **1** | 去掉内核里对 `DH` 的写死（`QSTR`、`BT*DH` 偏移） | dh=64 与改前**逐位一致** | ✅ 已完成（`f33a56e`，16 数组最大差 0.0）|
| **2** | **dh=128 + fp16 存储 + `BQ=32, BK=128` tile** | 与物化路径相对误差 ≤1e-3；`attn_sdpa_only` 明显下降 | 待做 |
| **3** | dh=96（fp16，覆盖常见配置） | 同上 | 待做 |
| **4** | 接进 `scaled_dot_product_attention` 自动选择；fp32 且 dh>64 时保留物化兜底 | 用户代码不改即生效 | 待做 |

**为什么第 2 步必须是 fp16**：dh=128 的 fp32 在任何实现里都没有快路（连 torch 都是物化），
所以照抄"fp32 分块"只会白干。fp16 还有另一个作用：**smem 减半**，让 `BQ=32, BK=128` 塞进 96 KB。

**预期收益**：若注意力从 30.5 ms 降到 torch 量级（个位数 ms），
整步 **49.7 → ~23 ms（约 2.1×）**，即在 5,667 tok/s 的基准上反超；
同时省掉 335 MB×2 的物化，显存也跟着降（那正好是"显存偏胖"的一块）。

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
