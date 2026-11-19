# seq2048：ORV 比 PyTorch 慢 130 ms —— 归因报告

> 口径（两侧相同）：V3 MoE d=1024 / d_ff=3328 / E=4+1 shared / top_k=2 / n_q=16 /
> n_kv=2 / csa=4（3 global + 9 CSA）/ vocab=96000 / seq=2048 / batch=1 / 12 层 /
> AMP 开（ORV `--amp gemm` + `--opt int8gpu`；PyT autocast fp16 + GradScaler）/
> ckpt 关 / 20 步、跳过预热与最初 2~3 步 / V100-PCIE-32GB / 驱动 580.88。
> ORV 0.9.97（远端 `D:\orv_pip_0962`，train/ 下 64/67 个 .py 与本地 HEAD md5 一致；
> 仅 `apilist.py`/`dist.py` 不同，均不在热路径）。

## 0. 先说结论（与任务给的 130 ms 对不上）

| 口径 | ORV | PyTorch | 差 |
| --- | --- | --- | --- |
| 今日实测，官方协议（预热 1 进程 + 3 跑取中位） | **0.490 s**（0.487/0.492，极差 0.9%） | **0.412 s**（0.410/0.414，极差 0.9%） | **78 ms**（PyT 快 1.19×） |
| 单次 sync + event 分相（同法量两侧） | 0.497 s（含 8 event/步 ≈ 7 ms 打点） | 0.400 s | 90~97 ms |
| 任务里给的基线 | 0.541 s | 0.411 s | 130 ms |

**0.541 s 今天复现不出来**：同一 `bench_stable.py`、同一机器，4 次独立进程都是
0.487–0.492（极差 0.9%）。今天的 ORV 稳态峰值 live = 23.708 GiB，而池 cap 是
19.12 GiB（`cap_bytes=20526582988`）—— 与任务里"自适应 cap 19.1 GiB"一致；
0.541 很可能是在显存更紧（cap 更低/有别的进程占卡）时测的。**本报告全部按今天的 490 ms 口径给。**
→ 130 ms 的真身是 **78 ms（协议口径）/ 90 ms（物理口径）**。

关键结论一句话：**两侧的 GPU kernel 时间几乎一样（367.0 vs 353.3 ms/步），
差的是"GPU 空转"：ORV 130 ms/步、PyT 46 ms/步。**

---

## 1. ORV kernel 榜（top 25）

命令：`prof_event_kernels.py`（每 leaf 一对 CUDA event，步末批量读，不 sync），
seq2048 / 17 步稳态 / wall 572.8 ms（含打点 83 ms，无打点同进程 491 ms）。

叶子（逐 kernel event）合计 **367.0 ms/步**，leaf 调用 **6720 次/步**。

| # | leaf / kernel | ms/步 | %墙钟 | %叶子 | 前向 | 反向 | 次/步 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cublasGemmEx_fp16 | 88.34 | 15.4% | 24.1% | 29.9 | 58.5 | 555 |
| 2 | cublasSgemmStridedBatched（**fp32**） | 72.61 | 12.7% | 19.8% | 22.9 | 49.7 | 369 |
| 3 | cast_f32_to_f16 | 29.88 | 5.2% | 8.1% | 11.2 | 18.7 | 929 |
| 4 | adamw_multi_m8v16 | 22.46 | 3.9% | 6.1% | – | – | 1 |
| 5 | ewise_scale_v4 | 15.97 | 2.8% | 4.4% | 3.9 | 2.2 | 465 |
| 6 | mul_bcast8 | 12.51 | 2.2% | 3.4% | 5.1 | 7.4 | 576 |
| 7 | softmax_lastdim_bwd | 9.28 | 1.6% | 2.5% | – | 9.3 | 33 |
| 8 | softmax_lastdim | 8.71 | 1.5% | 2.4% | 8.7 | – | 33 |
| 9 | fused_add_ip_v4 | 8.51 | 1.5% | 2.3% | – | 8.5 | 348 |
| 10 | moe_scatter_pairs | 8.29 | 1.4% | 2.3% | – | 8.3 | 12 |
| 11 | sumsq_partial | 7.63 | 1.3% | 2.1% | – | – | 321 |
| 12 | situ_glu_bwd | 6.72 | 1.2% | 1.8% | – | 6.7 | 60 |
| 13 | swap_last2 | 6.67 | 1.2% | 1.8% | 0.8 | 5.9 | 135 |
| 14 | sum_axis8 | 6.43 | 1.1% | 1.8% | 1.1 | 5.3 | 138 |
| 15 | ewise_add_v4 | 5.83 | 1.0% | 1.6% | 5.8 | – | 129 |
| 16 | ce_bwd | 4.35 | 0.8% | 1.2% | – | 4.3 | 1 |
| 17 | situ_glu_fwd | 4.13 | 0.7% | 1.1% | 4.1 | – | 60 |
| 18 | transpose2d | 3.99 | 0.7% | 1.1% | 2.1 | 1.8 | 114 |
| 19 | gather_rows | 3.49 | 0.6% | 1.0% | 2.2 | 1.3 | 112 |
| 20 | mem:cuMemsetD32Async | 3.35 | 0.6% | 0.9% | 0.5 | 2.9 | 244 |
| 21 | scatter_add_rows | 3.33 | 0.6% | 0.9% | 1.4 | 1.9 | 112 |
| 22 | mem:cuMemcpyHtoDAsync | 3.29 | 0.6% | 0.9% | 3.1 | 0.1 | 220 |
| 23 | cat_place | 3.27 | 0.6% | 0.9% | 3.3 | – | 114 |
| 24 | index_gather8 | 3.07 | 0.5% | 0.8% | 3.1 | – | 94 |
| 25 | add_bcast8 | 2.99 | 0.5% | 0.8% | 3.0 | – | 57 |

其余 24 个 leaf 合计 21.89 ms（6.0%）。共 49 个不同 (kernel, shape) 组合。

### 四段（`PROF_LEAVES=0` clean 模式：每步只 8 个 event，wall 497.2 ms）

| 阶段 | 段 inclusive | 段内 kernel | 段内空转 | 空转占比 |
| --- | --- | --- | --- | --- |
| forward | 176.40 | 120.06 | **56.34** | 31.9% |
| backward | 214.27 | 201.81 | 12.46 | 5.8% |
| clip | 25.23 | 22.60 | 2.63 | 10.4% |
| optimizer | 22.47 | 22.48 | −0.02 | ~0 |
| 四段合计 | 438.37 | 366.95 | 71.42 | 16.3% |
| **段外** | **58.85** | – | 58.85 | – |
| 整步 | 497.23 | 366.95 | **130.28** | 26.2% |

- **前向 56 ms 空转与 MoE 路由的 12 次阻塞式 D2H 吻合**：`v3_moe.route_topk` 每层一次
  `idx_buf.to_numpy()`（12 层），该路径 `cuMemcpyDtoHAsync + cuStreamSynchronize`
  实测每次主机阻塞 **4.18 ms**（26 次/步合计 108.6 ms 主机时间）；12 × 4.18 = 50.2 ms，
  与前向空转 56.3 ms 相差 6 ms（差值是同一段里的分配/其它 D2H）。注：26 次里另 14 次
  分布在 loss/gnorm 标量回读与层内其它 host 回读，**未逐次打 phase 标签**。
- **段外 58.9 ms**：`opt.zero_grad()` 释放梯度 + `rt.f16_cache_clear()` 释放 321 份
  fp16 权重影子 + 2 次标量回读（loss / gnorm，`cuCtxSynchronize` 各 13.6 ms，合计 27.3 ms）
  + `corpus.take`。**同时段主机侧 `cuMemFree` 1333 次/步、65.3 ms**。
- 显存池在抖动：20 步进程累计 **misses=24448（≈1225 次/步）**、**trims=2769（≈138 次/步）**，
  `cuMemAlloc` 1225 次/步 = 57.5 ms 主机时间（其中前向 47.2 ms）。

---

## 2. PyTorch 类别表

`torch.profiler(activities=[CPU,CUDA])`，20 步。**必须取"叶子事件"的 `self_device_time`**：
torch 的 range 归因会把同一段 kernel 时间同时记到父 range 和子 op 上（实测全事件求和
665.9 ms vs 叶子 366.0 ms，重复 1.8×）。

- 墙钟（单次 sync，中位）：**399.6 ms**（0.395–0.408，极差 3.2%）
- CUDA kernel（叶子）合计：**353.3 ms/步** ⇒ **CUDA/墙钟 = 0.884**
- kernel 次数 25064/步（60 个不同 op）；CPU aten 事件 51615/步
- 分相（event）：fwd 107.5 / bwd 206.2 / **opt 85.9** ms（三段和 399.6 = 墙钟）

| 类别 | ms/步 | %CUDA | %墙钟 | 次/步 |
| --- | --- | --- | --- | --- |
| 1_GEMM（aten::mm/bmm） | 94.01 | 26.6% | 23.5% | 811 |
| 3_EW_NORM（mul/add/div/tanh/sum…） | 115.82 | 32.8% | 29.0% | 5308 |
| 4_MOVE_MOE（copy_/as_strided/fill_/cat/empty…） | 107.67 | 30.5% | 26.9% | 15874 |
| 5_CE_CLIP_OPT（_foreach_*/GradScaler/log_softmax） | 22.47\* | 6.4% | 5.6% | 4 |
| 9_OTHER | 13.35 | 3.8% | 3.3% | 3067 |

\* 这一行是**下限**：`_foreach_addcdiv_/addcmul_/add_/mul_/div_/sqrt` 六个 op 在本次
profiler 里带 cpu_children，它们的 kernel 被记到更内层的 aten 子 op（如 `aten::result_type`、
`aten::empty`）上，所以叶子归属把它们算进了别的类。**分相独立窗口**实测 optimizer 段的
kernel 合计是 **31.80 ms**（`_foreach_lerp_` 11.26 + `result_type` 8.27 +
`_amp_foreach_non_finite_check_and_unscale_` 7.89 + …），下表与墙钟分解用的是 31.8。

分相 kernel 明细（`prof_pyt_phases.py`，单相独立窗口，含 ~5-10% profiler 抬高）：
fwd 114.1 / bwd 207.1 / **opt 31.8** ms。⇒ **PyT 的 optimizer 段 85.9 ms 里只有 31.8 ms 是 kernel，
54 ms 是主机停顿**（`_foreach_*` 的 CPU 时间 + GradScaler 的非有限检查同步）。

---

## 3. 分组对照 + 差距分解

### 3.1 同一套类别（kernel 时间）

| 类别 | ORV ms | PyT ms | 差（ORV−PyT） | 倍数 |
| --- | --- | --- | --- | --- |
| 1_GEMM | 161.76 | 94.01 | **+67.75** | 1.72× |
| 2_ATTN（ORV 独立 softmax；PyT 已融合） | 17.99 | 0.00\* | +17.99 | n/a |
| 3_EW_NORM | 73.29 | 115.82 | **−42.53** | 0.63× |
| 4_MOVE_MOE | 74.87 | 107.67 | **−32.79** | 0.70× |
| 5_CE_CLIP_OPT | 39.06 | 22.47\* | +16.59 | 1.74× |
| 9_OTHER | 0.00 | 13.35 | −13.35 | – |
| **kernel 合计** | **366.97** | **353.31** | **+13.66** | 1.04× |
| 墙钟（单次 sync 口径） | 490 | 400 | +90 | 1.23× |
| **非 kernel（墙钟−kernel）** | **123** | **46** | **+77** | 2.7× |

\* PyT 的 mem-efficient attention 是 non-leaf range（`aten::_efficient_attention_*`，
实测 0.3~2.6 ms/次），其 kernel 被下属 aten op 承接，落在 GEMM/EW/MOVE 里；
PyT 侧独立的 softmax 只有 `aten::_softmax` 0.28 ms + CE 的 `_log_softmax` 3.95 ms。
**所以 2_ATTN 这一行两侧不可直接比**：ORV 把注意力拆成「fp32 bmm（记在 1_GEMM）+ 独立 softmax 核」，
PyT 合成一个 fp16 融合核。

### 3.2 按"墙钟"分解（才是 78/90 ms 的真身）

| 段落 | ORV | PyT | 差 | 其中 kernel 差 | 其中空转差 |
| --- | --- | --- | --- | --- | --- |
| forward | 176.4 | 107.5 | **+68.9** | +12.6 | **+56.3** |
| backward | 214.3 | 206.2 | +8.1 | −5.3 | +12.5 |
| clip | 25.2 | – | **+25.2** | +22.6 | +2.6 |
| optimizer | 22.5 | 85.9 | **−63.4** | −9.3 | −54.1 |
| 段外（zero_grad/标量回读/清缓存/取数据） | 58.9 | 0（含在段内） | +58.9 | – | +58.9 |
| 合计 | 497.2 | 399.6 | +97.6 | +13.7 | +83.9 |

（ORV 侧 497 含 ~7 ms 打点；扣掉后物理差 ≈ 90 ms；官方 4-sync 协议下 PyT 变 412 ⇒ 78 ms。）

### 3.3 130 ms 的"前三名"

- **按 kernel 类别**：1_GEMM **+67.75 ms**、3_EW_NORM **−42.53 ms**、4_MOVE_MOE **−32.79 ms**
  （净和只有 +13.7 ms —— kernel 侧基本互相抵消）。
- **按墙钟**：forward 的 GPU 空转 **+56 ms**、段外 **+59 ms**、clip **+25 ms**；
  被 optimizer **−63 ms** 抵掉一大截。

⇒ 不能只报 kernel 类别：**130(今 78~90) ms 里 84% 是空转/同步/分配，16% 是 kernel 时间。**

---

## 4. 定性判据

### 4.1 贡献最大的 kernel 类别（1_GEMM，+67.8 ms）→ **(b) 同类 kernel 在该尺寸下效率更低**

同一份数学（两侧同构，Σ2mnk·batch·次数 = **6.449 TFLOP/步**），ORV 内部却分成两条精度路径：

| ORV cuBLAS 入口 | 次/步 | ms/步 | TFLOP/步 | 有效 TFLOPS |
| --- | --- | --- | --- | --- |
| cublasGemmEx（**fp16**） | 555 | 88.34 | 5.730 | **64.9** |
| cublasSgemmStridedBatched（**fp32**） | 369 | 72.61 | 0.719 | **9.9** |
| cublasSgemm_fp32 | 24 | 0.82 | 0.000 | 0.5 |
| 合计 | 948 | 161.76 | 6.449 | 39.9 |

- 词表 96000 的 head GEMM、MoE 专家 GEMM 走 fp16，**64.9 TFLOPS**（PyT 同口径 68.6，持平）
- 但 **11% 的 FLOPs 吃了 45% 的 GEMM 时间**：369 次 fp32 strided-batched（9.9 TFLOPS ≈ fp32 峰值的 63%）
- **源码级根因**（不是猜）：`cuda.py:5851 bmm()` → `sgemm_strided_batched()`（5884）→
  `cublasSgemmStridedBatched`，**只有 fp32 一条路**，缓冲区按 4 B/元素算 step；
  而 AMP 的 fp16 路径只挂在 2-D 的 `A @ B.T`（`__init__.py:1029-1045` `gemm_f16_*_raw`）。
  调用点探针（`probe_gemm_sites.py`）确认：
  - 注意力打分 `sgemm_nt_strided_batched ← _sdpa_impl:3106 → _sdpa_materialized:3463 → _attn_scores:3389`
  - 注意力 dQ `sgemm_tn_strided_batched ← __init__.py:3397`
  - 3-D `@` 走 `sgemm_strided_batched ← __init__.py:1010 _matmul_cuda`
  - 另有 MoE 反向的部分 GEMM 落到 `sgemm`（`__init__.py:1074/1085` 的 fp32 回退）
- 量化收益：把 0.719 TFLOP 按 65 TFLOPS 跑 ⇒ 11.1 ms（现在 72.6）⇒ **−61 ms kernel**

### 4.2 空转那 84%（forward 56 ms + 段外 59 ms）→ **(c) ORV 有 PyT 没有的额外开销（同步/驱动调用）**

| 项目 | ORV（实测/步） | PyT（实测/步） |
| --- | --- | --- |
| 阻塞式 D2H（`cuMemcpyDtoHAsync`+`cuStreamSynchronize`） | **26 次**，主机阻塞 108.6 ms（4.18 ms/次） | 无同类；`aten::nonzero` 228 次（主机 74.9 ms）、`aten::item`/`_local_scalar_dense` 745 次（17.6 ms）——**PyTorch 靠 C++ 深队列把这些等待藏在 GPU 忙时** |
| 逐 step 驱动分配 | **`cuMemAlloc` 1225 次（57.5 ms）+ `cuMemFree` 1333 次（65.3 ms）** | 缓存分配器；本步未见 cudaMalloc（**未实测**，仅从 profiler 无相关记录推断） |
| 显存池 trim | 138 次/步（cap 19.12 GiB < 峰值 live 23.71 GiB） | 不适用 |
| 整设备同步 | `cuCtxSynchronize` 2 次/步（27.3 ms，loss/gnorm 标量回读） | 无（只有 `.item()` 已在上面计） |
| 独立 softmax 核 | 33 次/步，17.99 ms | 融合进 mem-efficient attention |

### 4.3 被任务点名的 cast → **不是 (c)，反而是 (a) 的反例**

| | ORV | PyT |
| --- | --- | --- |
| cast/拷贝类 op | **929 次/步**（`cast_f32_to_f16`），29.88 ms | `aten::copy_` 2537 + `_to_copy` ~1563 次/步，56.28 ms |
| 流量/效率 | **18.21 GB/步，加权有效 609 GB/s**（V100 STREAM ≈ 830） | 未按字节拆分 |
| 大张量单次效率 | 3.4M 元素 592 GB/s；196.5M 元素 **781 GB/s**；98.3M 元素 758 GB/s | – |

⇒ ORV 的 cast **次数更少、单核效率 0.73~0.94×STREAM**，「每步 860 次 cast 是额外开销」这个假设**不成立**；
真正额外的是 4.2 的同步/分配。

---

## 5. 下一步最该改的一处

**把 3-D 批量 GEMM（注意力与 3-D `@`）接进 AMP 的 fp16 路径** ——
即给 `bmm`/`sgemm_strided_batched` 增加 `cublasGemmStridedBatchedEx(CUDA_R_16F)` 分支
（fp32 累加、fp32 输出），与现有 `gemm_f16_*_raw` 同一套契约。

- **预期收益**：那 369 次 fp32 调用从 72.6 ms → ~11-15 ms ⇒ **−55 ~ −60 ms/步**（0.490 → ~0.43，追平 PyT 的 0.412）。
- **风险**：① 注意力打分改 fp16 后 softmax 输入范围/精度变化（QK-norm 已在，但仍需 loss 曲线对拍）；
  ② 需要 fp16 的 Q/K/V scratch（复用 `_fp16_scratch`，显存 +~0.3 GB）；
  ③ PyT 的 autocast 本来就跑 fp16 注意力，所以这是"对齐语义"而不是"新语义"。

**次优先（更便宜、可当第一枪）**：把自适应池 cap 从 19.12 GiB 抬到 ≥ 峰值 live（如 `TG_POOL_GB=26`）。
文档实测曲线 12→0.775 / 16→0.694 / **20→0.501 s**，而当前 19.12 GiB 仍低于 23.71 GiB 峰值 live
⇒ 仍有 1225 次 alloc + 1333 次 free + 138 次 trim/步。预期 **−25 ~ −50 ms**，风险是 32 GB 卡的 OOM 余量（需 A/B）。
第三：MoE 路由去掉 12 次 D2H（需把 dispatch 改成设备端），预期 −40~−56 ms，但改动最大。

---

## 6. 新建文件与未验证项

### 新建脚本（全部在 `lab/rent/`，未改 `python/tgai_go/`）

本地：`x.py`、`probe_remote.py`、`dump_md5.py`、`cmp_md5.py`、`probe_procs.py`、`probe_cmdline.py`、
`run_prof_one.py`、`show_orv.py`、`show_phases.py`、`analyze_gap.py`、`deep_dive.py`、
`prof_pyt.py`、`prof_pyt_phases.py`、`probe_prof_schema.py`、`probe_attn_leaf.py`、
`probe_gemm_sites.py`、`ref_pyt/{v3_pyt_train.py,model_v3.py}`、数据 `prof_2048_new.json`、
`prof_2048_phases.json`、`prof_pyt_2048.json`、`remote_md5.json`。
远端（`D:\orv_bench\`）：同名副本 + `prof_2048_vanilla.json`、`prof_pyt_phases.json`。

### 未验证 / 需要注意

1. **PyT 侧 kernel 绝对时间**来自被 profiler 拖慢的 run（墙钟 1087 ms/步 vs 正常 400）。
   kernel 时长由 CUPTI 用 GPU 时间戳记录，但单核单独运行、L2 冷，估计 +5~10%。
   本报告对 PyT 只用它的**分布与合计 353 ms**，并与 ORV 的 367 ms 同法比较。
2. **PyT 每步 cudaMalloc 次数未实测**（只用缓存分配器语义推断为 0）。
3. **两侧模型不完全同构**：ORV 848.645M 参数（**head 不与 embed 共享**），PyT 740.769M
   （`tie_embeddings=True`）；PyT 还多 `attn_res`（ORV 无），ORV 多 `clip_grad_norm_`（PyT 无）
   与 int8 优化器（PyT 是 fp32 foreach AdamW）。逐类别对比时这些差异仍存在。
4. **0.541 s 基线复现失败**（今天 0.490）；本文所有绝对数按今天口径。
5. 只测了 seq2048，未重测 seq512（任务要求同口径总表时以 2048 为主）。
6. `2_ATTN` 行两侧不可直接比（PyT 融合注意力不产生独立 softmax kernel）。
7. 显存池 cap 与 trims 的因果关系是**推断**（cap<live 时每次 release 触发 trim 是源码 4238-4249 行
   写明的机制），未做「只改 cap」的 A/B。
