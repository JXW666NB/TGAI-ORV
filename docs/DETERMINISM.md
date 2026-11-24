# 非确定性全清单 与 `TG_DETERMINISTIC`

> 2026-11-20。本文回答两件事：**哪些地方会让同一条命令两次运行结果不同**，
> 以及 **`TG_DETERMINISTIC=1` 覆盖了哪些、没覆盖哪些**。
> 数字与判据都来自本机实测（V100-32G），不是推测。

## 0. 一句话结论

- **默认非确定**：同命令、同 seed、同数据跑两次，step-0（纯前向）恒等，
  **step-1 起 loss 在最末几位分叉**（实测 rel 1.3e-06）。
- `TG_DETERMINISTIC=1` 之后，`v3_pretrain` 的 20 步 loss/gnorm 在**两个独立进程**间
  **逐位相同**（4 组配置实测：seq2048 gemm/int8gpu、seq2048 amp-off、seq2048 adamw、seq512）。
- 非确定性有**两处**，不是一个：embedding 散射（`scatter_add_rows`）**和**
  RMSNorm 的 dw（`rmsnorm_bwd` 的 `atomicAdd`）。**只修 embedding 会得到假的安全感** ——
  旧包在开关打开时仍有 25/321 个参数（全是 `ln1/ln2/ln_f`）每次运行都在变。

## 1. 开关语义

```bash
TG_DETERMINISTIC=1     # 逐位可复现（慢一点）；默认不设 = 现状（快）
TG_DETERMINISTIC=0     # 显式关闭
```

也接受 `true/false`、`on/off`、`yes/no`（大小写不敏感）。**其它值直接抛
`CudaError`**，不静默当关 —— 静默降级会让"我明明设了"变成查不出来的假象。

程序里等价于 `T._cuda().deterministic = True`（`CudaRuntime` 构造时从环境变量读一次）。

## 2. 全清单（位置 / 机制 / 是否影响数值 / 是否被开关覆盖）

### 2.1 CUDA 内核里的原子操作

| 位置（`python/tgai_go/train/`） | 机制 | 影响数值？ | 被开关覆盖？ |
| --- | --- | --- | --- |
| `cuda.py: rmsnorm_bwd` | `atomicAdd(dw+i, …)` 跨 **2048 行**累加，同一列被多 block 抢着加 | **是**（`ln1/ln2/ln_f` 梯度 1e-9 量级漂移） | ✅ 开关下 `rmsnorm_bwd` 分派到 `rms_rstd + rmsnorm_bwd_dx + rms_dw_det`（列独占、行号升序） |
| `cuda.py: emb_gather_bwd` | embedding 反向，**重复 token id** 时同一行被多 block 抢着加 | **是** | ✅ 走 `emb_bwd_owned`（`id % grid` 独占、token 升序）；**已去掉** `vocab>=2 && d>=128` 门槛 |
| `cuda.py: scatter_add_rows` | 行 gather 反向（`P["emb"][ids]`、MoE `index_select` 的反向） | **是** | ✅ 走 `scatter_rows_owned` |
| `cuda.py: scatter_add_flat` | 逐元素 gather 反向（`gate[order]` 等） | **是**（重复下标时） | ✅ 走 `scatter_flat_owned` |
| `cuda.py: index_scatter_add8` | 切片 / `cat` / `repeat_interleave` 反向 | **切片与 cat 不影响**（源位置互不重叠 ⇒ 每个地址只加 1 次，顺序无关）；**`divs>1`（repeat）会重叠 ⇒ 影响** | ⚠️ 部分：`repeat_interleave` 在开关下改走 `repeat_reduce`（按 j 升序）；切片本身不需要覆盖 |
| `cuda.py: moe_scatter_rows` | 按专家分段把 `(token,slot)` 的梯度散回 token 行 | **否**：top-k 每个 token 在每个专家里最多出现一次 ⇒ 每次发射每个地址只加 1 次 | 不需要（仍走原子，结果确定） |
| `cuda.py: moe_permute` | `atomicAdd(&counts[e],1)` 领槽位号 | **是**：槽位分配顺序随调度 ⇒ 超出 capacity 时**被丢弃的行集合**可能不同 | ❌ **未覆盖**（只被 CUDA Graph 的 MoE 路径使用；`v3_pretrain` 不走） |
| `cuda.py: moe_permute_bwd` | `atomicAdd(dx[row*d+c], dy)`（同 row 的 2 个 slot 撞） | 2 个加数可交换 ⇒ 本身不影响；但受上一行"被保留的行集合"影响 | ❌ 未覆盖（同上） |
| `cuda.py: emb_bwd_hist` | `atomicAdd(int* counts, 1)` | **否**：整数原子加法与顺序无关 | 不需要 |
| `cuda.py: moe_scatter_pairs` / `moe_compact` / `moe_scatter_slots` / `sumsq_sequential` | 单线程串行（blockIdx/threadIdx==0） | **否**：顺序固定 | 不需要 |

### 2.2 归约顺序（多 block 归约 / 线程内串行）

| 位置 | 机制 | 影响数值？ |
| --- | --- | --- |
| `blk_sum` 树归约（`rmsnorm_fwd`/`rmsnorm_bwd`/`rmsnorm_bwd_dx`/`attn_mask_softmax`/`ce_row_stats`） | 固定 `tid → 数据` 映射 + 固定树（`s = blockDim>>1 … 1`） | 否 |
| `softmax_lastdim` / `_bwd`、`log_softmax_lastdim` / `_bwd` | warp `__shfl_xor` 固定 XOR 顺序（16,8,4,2,1），lane 映射固定 | 否 |
| `sum_axis8` / `sum_axis0_rows` | 每个输出元素**一个线程按 index 升序串行** / 固定树 | 否 |
| `ce_fwd_rows` + `ce_fwd_mean` | 每行一个 block 固定树；均值 = 单线程按行号升序 | 否 |
| `reduce_sum` / `reduce_sum_partial`+`reduce_sum` / `sumsq_partial`+`sumsq_final` | grid 只由 `n` 决定（不随数据变），树顺序固定 | 否 |
| `sumsq_multi_slots`（grad-norm 融合） | 每个张量一个 block 写自己的槽位，**刻意不用原子** | 否 |
| `repeat_reduce` | 每个输出元素一个线程，`j` 升序累加 | 否 |
| `rms_dw_det` | 每列一个线程、行号升序 | 否 |
| `emb_bwd_owned` / `emb_bwd_rows` / `emb_bwd_rows4` | 每个 id 独占、token 下标升序（`float4` 版同序） | 否 |
| `adamw_multi_int8` / `adamw_multi_m8v16` 的块内 absmax | warp `__shfl_xor` 固定顺序 | 否 |
| 所有逐元素内核（`ewise_*`、`add_bcast8`、`mul_bcast8`、`cast_*`、`fused_add_ip*` …） | 无归约 | 否 |

### 2.3 Python 侧

| 位置 | 机制 | 影响数值？ |
| --- | --- | --- |
| `set()` / `dict` 迭代（`seen`、`_params`、`_pool_free`、`GROUPS`、`alias` …） | 一处是**去重后 sorted**（`apilist`、`dispatch`、`ops_plugin`）；其余是"插入顺序"的 dict，同一次构建序列恒定；`_topo` 的 DFS 顺序来自 `_parents` 元组顺序 | 否 |
| `np.argsort` | MoE 分组 / top-k 全用 `kind="stable"`；`__init__.py` 的 `argsort(axes)`/排序算子也是同输入同输出 | 否 |
| 反向图拓扑 | `_topo` 用显式 stack + tuple 顺序 DFS，梯度累加顺序固定 | 否 |
| 数据顺序 | `PackedCorpus`：`seed + epoch*100003 + shard*7919`，分片内 `default_rng` 置换 | 否（同 seed 恒等） |
| `_CE_SEL_CACHE` / `h2d(cache_slot=…)` / fp16 权重影子 | 只影响"是否重算"，不影响值 | 否 |
| `hash()` | 热路径没有用（grep 无命中） | 否 |

### 2.4 驱动 / 环境

| 项 | 结论 |
| --- | --- |
| cuBLAS GEMM 算法选择 | 固定形状/固定指针 ⇒ 同一次运行的算法不变。实测 6 个真实形状（fp32 + fp16，含 `TN 1024×1024×2048`、`NT 16×2048×2048×64`）**连跑 3 次 max_abs_diff = 0.0**。**但这不是全局保证**：其它形状若命中 split-K + 原子归约就会变；本开关**不**设置 `CUBLAS_WORKSPACE_CONFIG` |
| 显存池地址 | 池复用会让指针逐次不同，但只影响对齐，不影响数值 |
| Windows 计时粒度 | 15.6 ms，**不能**用 `time.time()` 判性能；本文所有性能数字用 CUDA event |

## 3. 判据（怎么自己验）

```bash
# 1) 逐位复现（两个独立进程 × 20 步，比较每步 loss/gnorm 的 repr）
#    实现在 lab/rent/det/det_batch.py（spec 里给 pkg/env/seq/amp/opt）
TG_DETERMINISTIC=1 python det_batch.py --spec spec_phase3.json
#    -> new_on_gemm  loss_bitwise=True gnorm_bitwise=True

# 2) 梯度级：同进程两次 forward+backward，逐参数比梯度
python det_grad_probe.py <冻结包目录> det
#    -> params_differing=0/321

# 3) 开/关同一步的数值差
python det_grad_diff.py <冻结包目录>
#    -> loss_equal=True, global_rel_err=9.24e-09, 25/321 参数不同（全是 RMSNorm dw）

# 4) 算子级代价（CUDA event，真实形状）
python det_perf_kernel.py <冻结包目录>
```

`tests/test_deterministic.py` 是仓库内的门禁（GPU-gated，`TG_RUN_CUDA_TESTS=1`），
它只覆盖 `qwen` 路径的 `emb_bwd`/`rms_dw_det`/`repeat_reduce`；`v3_pretrain`
的端到端可复现性用上面第 1 条验。

## 4. 开关保证不了的事

1. **`moe_permute` / `moe_permute_bwd`**（MoE 的 CUDA Graph 路径）：槽位领号是原子的，
   超容量时丢弃哪些行随调度变。`v3_pretrain` 不走这条；要用它 + 逐位复现，得先把
   领号改成确定性分配。
2. **cuBLAS 自身**：没有强制 workspace/算法，极端形状下 split-K 仍可能引入原子归约。
3. **跨进程/跨机器的浮点一致**：本文的"逐位"是**同一台机器、同一驱动、同一二进制**内的
   两次运行；不同 GPU 型号/驱动的 `__expf`、`rsqrtf`、cuBLAS 内核不同，不保证逐位。
4. **多卡 `dist.py`**：归约走主机中转（按 rank 顺序），参与的是已经确定的梯度；
   本开关不改变它的行为，也未在真多卡上验证过逐位（单卡两 rank 验证过一致性）。

## 5. 实测（2026-11-20，V100-32G，`v3_pretrain` d=1024/L=12/E=4/top_k=2/csa=4）

| 项 | 结果 |
| --- | --- |
| 开，seq2048 `--amp gemm --opt int8gpu`，2 进程 × 20 步 | loss / gnorm **逐位相同** |
| 开，seq2048 `--amp off` / `--opt adamw` / seq512 | 各自 2 进程 × 20 步 **逐位相同** |
| 旧包 + 开，seq2048 | **仍分叉**（step-1 rel 7.9e-06）⇒ 旧开关盖不住 |
| 关（冻结原包）2 进程 × 20 步 | 分叉（step-1 rel 9.8e-07）—— 默认行为未被改动 |
| 前向 loss（开 vs 关） | **逐位相同**（`11.669746398925781`） |
| 梯度（开 vs 关） | 全局相对误差 **9.24e-09**；25/321 参数不同，max\|Δ\| 7.9e-09 |
| 算子级代价 | embedding 散射 +0.61 ms、MoE gather 反向 +1.24 ms、RMSNorm 反向 +0.30 ms（真实形状，CUDA event） |
| 端到端代价 | 两次会话结论相反（+21.3% / −3.2%）；同一二进制不同时段步时本身漂移 0.219→0.277 s（+26%）⇒ **低于噪声底，无法可靠分辨** |
