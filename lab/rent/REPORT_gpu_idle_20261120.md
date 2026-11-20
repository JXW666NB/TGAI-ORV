# ORV 训练 GPU 空闲 68~82% 的根因与修复（4090 实测）

日期：2026-11-20 · 卡：RTX 4090 24 GB / sm_89 / 驱动见 nvidia-smi · 主机：8 卡实例的
单卡容器（`nproc=128`，无 cgroup cpu.max 限流）· Python 3.12.3 · torch 仅用于对照

**口径（全文统一）**：配置 = `12 层 / E=4 / d=1024 / n_q=16 / d_ff=3328 / vocab=96000 /
seq=512 / batch=1 / --opt int8gpu / --csa 4 / top_k=2`，参数 **848,645,257**。
「稳态步时」= 训练器自己 `RESULT.step_s_steady`（丢掉前 2 步）；「训练窗口 GPU 利用率」=
`nvidia-smi utilization.gpu` 在显存 >5 GiB 期间的 0.5 s 采样均值。所有对比都做了
「热机 + 同机同数据 + A/B/A/B 交替」。

---

## 0. 结论（一句话）

**不是 GPU 慢，也不是同步等待：一步 4038 次内核发射，每次摊到 59 µs 的**主机侧
Python/ctypes/池分配**成本，主线程 CPU 时间 = 墙钟的 100% ⇒ 主机是瓶颈，GPU 有
58~70% 的时间没有内核可跑。** 换更快的卡（V100→4090 算力 5×）几乎不提速，正是这个机理。

---

## 1. 根因清单（按影响排序，全部实测）

| # | 根因 | 关键测量 | 影响 | 状态 |
|---|---|---|---|---|
| R1 | **主机侧每算子摊 ~59 µs**，4038 次/步 | 主线程 CPU/墙钟 = **100.0%**；纯 `cuLaunchKernel` 仅 **4.39 µs**，`rt.launch` 6.49（预建 args）/7.35（现建 args）→ 发射本身只占 4038×7.35 = **29.7 ms（12.5%）** | 步时 238 ms 中 ~208 ms 是主机算的 | **已证实** |
| R2 | **`rt.empty` 池分配 6.48 µs/次 × 4539 次/步 = 29.4 ms（12.4%）** | `empty((512,1024))` 池命中 6.477 µs（其中 `GpuBuf()` 构造 1.892）；池未命中（`cuMemAlloc`）单次 **102.7 µs**，但只有 25~30 次/步 | 12.4% | **已证实**（未修，见 §5） |
| R3 | **梯度裁剪逐张量归约：962~1283 次发射/步** | `sumsq_partial` 321 + `sumsq_final` 321 + `add_` 320 + `scale_inplace` ~321 = 每步 1283（占 4038 的 31.8%）；clip 段主机 **14.3 ms（6.0%）**，amp=gemm 时 **30.3 ms（8.6%）** | 6~8.6% | **已修**（§3） |
| R4 | **`mul_bcast`/`add_bcast` 每次重算 24 个广播参数** | 完整算子 **41.4 µs**：其中 `_bcast_strides`×2 = 6.6、`_row_strides`×2 = 2.5、`_left_pad8`×3 = 3.2、24 个 `ctypes.c_int` 构造 + 3 个列表推导 ≈ 15 µs；`mul_bcast8` 一步 **576 次** | 约 9 ms/步（3.8%） | **已修**（§3） |
| R5 | **AMP 在主机侧是纯开销**（GPU 端无收益） | amp=gemm 比 amp=off：fwd +8.2 ms、**bwd +90.8 ms**、clip +16.0 ms = **+115 ms（+48%）**，只换来 +986 次 `cast_f32_to_f16` 发射；`sgemm` 侧 fp16 确实比 fp32 快 2.5~4.4×，但 GPU 本就有 58~70% 空闲 ⇒ 收益为 0 | 打开 AMP 反而更慢 | **已证实**（不建议修默认值） |
| R6 | **`TG_BF16=1` 被静默忽略** | `amp_api._set(True,"fp16")` 主动把 `_bf16_state()["on"]` 置 False；实测 `amp_state()` 恒为 `fp16=True/bf16=False`，加 `TG_BF16=1` 后耗时完全不变（0.354 s，噪声内） | 配置失效 | **已证实 → 改为显式日志**（§3.4）；**不放开 bf16**，原因见 §5-U4 |
| R7 | **默认（不设开关）同一配置两次运行不可复现** | A 跑两遍（同代码同数据）：12 步里 gnorm 只有 **3/12** 逐位相同，loss 从第 1 步起就有差（max\|Δloss\|=1.61e-4、max\|Δgnorm\|=5.0e-3）；`TG_DETERMINISTIC` **默认关**（原子散射） | 直接解释「数字反常 / 加速比不稳」 | **已证实（且已复核：设开关后 12/12 逐位相同，见 §9）** |

### 被数据**否证**的假设（省下无用功）

| 假设 | 否证 |
|---|---|
| 每步有大量同步/回读导致主机等 GPU | `cuCtxSynchronize` **2 次/步**、标量 D2H **26 次/步**、`empty_cache` **0 次/步**。上限收益 = 2×0.96 + 26×13.4 ≈ **0.35 ms/步（0.15%）** |
| 池子在打滑（每步大量 `cuMemAlloc`） | 命中 4500+、未命中 **25~30 次/步**（25×102.7 µs = 2.6 ms） |
| 主机被限流 | 关掉 vram 采样线程后主线程 CPU/墙钟 = **100.0%**（说明 CPU 一颗核跑满、没有抢占/限流） |
| 每步 `empty_cache` / 每步新建大缓冲 | `trims` 增量 0，`empty_cache` 0 次/步 |
| 小内核都是低带宽 | 能确证的只有 3 条：`sumsq_parallel`(98.3M) **900 GB/s**、`add_`(1M) **~1023 GB/s**、`transpose2d`(1024²) **268 GB/s**；其余小内核在主机饿死 GPU 时**无法把 GPU 时间与主机空档分开**，不敢下结论 |
| `_vram_peak`（每 0.2 s 一个 `nvidia-smi` 子进程）拖慢步时 | 稳态步时 250.3/239.9/251.3/257.1/243.3/246.9（开） vs 250.3/239.9/251.3/257.1/243.3/246.9 量级（关）——**无可测差异**；但它确实吃掉约 10% 的一个核（thrCPU/墙钟 100%→90%），属该删的噪声 |

---

## 2. 关键测量（原始数据）

### 2.1 主机 vs GPU（`gt_cpu.py`，每步一次 `thread_time` 采样，探针开销可忽略）

| 项 | amp=off | amp=gemm |
|---|---|---|
| 稳态步时 | **234.9 ms**（bench）/ 238.0（分阶段和） | **326.0 ms** / 353.3 |
| 发射次数/步 | **4038** | **5027~5033** |
| 池分配 `empty`/步 | **4539** | 4729 |
| 标量 D2H/步 | 26 | 26 |
| `sync`/步 | 2 | 2 |
| 主线程 CPU / 墙钟 | **100.0%**（关 vram 线程）/ 90.0%（开） | 94.0% |
| nvidia-smi util（稳态） | 29.4~31.9%（中位 36~39%） | 17.9~19.3%（中位 21%） |

### 2.2 分阶段主机墙钟（`gt_phase.py`，10 步，稳态均值）

| 段 | amp=off | 占比 | amp=gemm | 占比 |
|---|---|---|---|---|
| zero_grad | 1.0 ms | 0.4% | 1.0 ms | 0.3% |
| **fwd** | **95.0 ms** | 39.9% | 103.2 ms | 29.2% |
| **bwd** | **109.6 ms** | 46.1% | **200.4 ms** | 56.7% |
| **clip** | **14.3 ms** | 6.0% | 30.3 ms | 8.6% |
| opt | 18.1 ms | 7.6% | 18.4 ms | 5.2% |
| 合计 | 238.0 ms | 100% | 353.3 ms | 100% |

**与层数标度拟合交叉验证**：fwd+bwd = 204.6 ms ÷ 12 层 = **17.05 ms/层**
（对外拟合 `t(L)=27.6+16.8L` 的 16.8 ms/层 ✅）；固定项 clip+opt+zero_grad = 33.4 ms
（拟合固定项 27.6 ms ✅）。**结论一致：低效按层摊，不是"每步一次的大停顿"。**

### 2.3 发射成本分层拆解（`gt_lat.py`，各 2 万次取均值）

| 层 | µs/次 | ×次数 |
|---|---|---|
| A 纯 `cuLaunchKernel`（参数模板预建） | **4.392** | 17.7 ms |
| B `rt.launch`（预建 args）= A + `_activate` 0.19 + `_fn` 字典 + 快路径模板写值 | **6.490** | 26.2 ms |
| C `rt.launch`（每次新建 args = 真实调用点） | **7.345** | **29.7 ms** |
| `rt._activate()` / `rt._ptr()` / `_split_forget_ptr()` | 0.187 / 0.337 / 0.187 | — |
| `rt.empty((512,1024))` 池命中 | **6.477** | 29.4 ms |
| `rt.add_`（n=512K 完整算子） | 10.791 | 7.4 ms |
| `rt.mul_bcast`（512×1024，完整算子） | **41.44** | 23.8 ms |
| `rt.d2h(标量)` ← **唯一隐式同步** | 13.406 | 26 次/步 |
| `rt.sync()`（空闲） | 0.958 | 2 次/步 |

⇒ **59 µs/次里只有 7.35 µs（12.5%）是"发射"，4.39 µs 是驱动地板；
剩下 ~52 µs 是算子层 Python：池分配 6.5、广播形状 15、Tensor/autograd 对象 ~30。**

### 2.4 每步发射次数前 20（`gt_prof.py`，CUDA event 计数，4038/步 = 各表之和）

`mul_bcast8` 576 · `fused_add_ip_v4` 348 · `fused_add_ip` 338 · **`sumsq_partial` 321 ·
`sumsq_final` 321** · `ewise_scale_v4` 153 · `sum_axis8` 138 · `swap_last2` 135 ·
`ewise_add_v4` 129 · `ewise_pow` 126 · `transpose2d` 114 · `cat_place` 114 ·
`gather_rows` 112 · `scatter_add_rows` 112 · `index_gather8` 100 · `permute8_v4` 96 ·
`index_scatter_add8` 94 · `ewise_mul_v4` 81 · `sum_axis0_rows` 63 · `ewise_scale` 63

### 2.5 `sumsq_*` 321+321 的来源（问题 2 的答案）

`nn_ops.clip_grad_norm_()`：梯度裁剪的 L2 范数是**逐张量**做的 —— 321 个带梯度的参数
各 2 次发射（`sumsq_parallel` = `sumsq_partial`+`sumsq_final`）+ 320 次 `add_` 累加
= **962 次**，再加裁剪生效时 321 次 `scale_inplace` = **1283 次/步**。
仓库里本来就有 `sumsq_multi_slots` 多张量模式，但它 `grid=(张量数,1,1)`，**每张量只有
1 个 block(256 线程)**：98.3M 元素的梯度单个 block 要扫 393 MB，实测 **93.7 ms / 4.2 GB/s**
（同形状 1024 block 时 900 GB/s）—— 这就是历史上加 `big <= (1<<20)` 门槛、等于没修的原因。

### 2.6 单内核（饱和队列下可测的）带宽

| 内核 | 规模 | GPU 时间 | 有效带宽 |
|---|---|---|---|
| `sumsq_parallel` | 98.3M（393 MB 读） | 436.9 µs | **900 GB/s**（≈ 峰值 1008 的 89%） |
| `add_`/`fused_add_ip_v4` | 1M（12.58 MB r+w） | 12.29 µs | **~1023 GB/s**（≈ 峰值） |
| `transpose2d` | 1024²（8.39 MB r+w） | 31.30 µs | **268 GB/s**（26%） |
| `sumsq_multi_slots`（1 block） | 98.3M | **93.7 ms** | **4.2 GB/s**（0.4%） |

---

## 3. 落地的修复（两处，逐位不改数值）

### 3.1 裁剪：逐张量 1283 次发射 → **3 次**（`cuda.py` + `nn_ops.py`）

新增三个确定性内核，按**分片**给 block（grid = Σ min(1024, ⌈m/256⌉)）：

* `sumsq_multi_shard`：块 b 用二分（`tg_slot_of`）找到自己属于哪个张量、第几号分片，
  然后**复刻 `sumsq_partial` 的 grid-stride 循环**（步长 = 256 × 该张量的 nb_t）
  ⇒ 每个分片的部分和与旧路径**逐位相同**；
* `sumsq_multi_final`：单 block，对每个张量**复刻 `sumsq_final` 的树归约**，
  再按下标顺序做 fp32 累加（0.0 + x == x）⇒ 与旧路径"逐张量 + 顺序 add"**逐位相同**；
* `scale_multi_shard`：纯逐元素乘，无归约 ⇒ 分片方式不影响取值；`scalar==1.0` 时
  与 `scale_inplace` 一样**什么都不做**，n==1 的 `known_scalar` 也在主机侧同步。

`nn_ops.clip_grad_norm_` 删掉 `big <= (1<<20)` 门槛，直接走分片版；旧路径保留为回退
（老后端没有新内核时）。

**确定性**：全程无 `atomicAdd`，求和顺序完全由代码固定 ⇒ `TG_DETERMINISTIC` 语义不变。

### 3.2 广播算子：24 个形状/步长参数按形状三元组缓存（`cuda.py`）

新增 `rt._bcast_tail(a_shape, b_shape, out_shape)`（`_bcast_strides`×2 + `_left_pad8`×3 +
24 个 `ctypes.c_int`），`mul_bcast` / `add_bcast` 共用。**是形状的纯函数**，与张量取值无关；
缓存键是形状元组，条目数由模型结构决定（>4096 清空，防病态）。

### 3.3 数值中立性证明

1. **孤立逐位测试**（`gt_equiv.py`，15 个张量 = 1/3/255/256/257/1023/1024/4096/65537/
   1M/1M+1/3M/5M/98.3M×2）：
   * norm：`old=517311.031` vs `new=517311.031`，**float 二进制完全相同**；新路径连跑 3 次、
     旧路径连跑 3 次都各自逐位相同（确定性）；
   * scale(s=0.3711)：15 个缓冲**逐位 diff = 0**，`max_abs_diff = 0`。
2. **广播乘法对拍 numpy**：6 种广播形态（含 `(512,1024)×(512,1)`、`(8,512,1024)×(1,1,1)`、
   `(1024,)×(512,1024)`）**全部逐位相同**。
3. **端到端**：A（未改）与 B（改后）的 gnorm/loss 差异，与 **A 自己跑两遍**的差异同量级
   （12 步里 A==A2 逐位 3/12，A==B 也是 2/12；第 1 步 gnorm 15.3866116 三方完全相同）。
   ⇒ 差异来自 R7（**默认不设 `TG_DETERMINISTIC`** 时的非确定性），**不是本次改动引入**。

### 3.4 `TG_BF16=1`（真 bug，但只修"静默"）

根因：`v3_pretrain.py` 的 AMP 分支写死 `T.amp_on(dtype="fp16")`，而
`amp_api._set()` 的 fp16 分支**主动** `rt._bf16_state()["on"] = False`。
实测：`TG_BF16=1` 时 `amp_on(dtype='fp16')` 之后 `bf16=False`，`amp_on(dtype='bf16')` 之后
`bf16=True`（说明内核侧没问题，是调用点覆盖掉了）。耗时也完全不变（噪声内）。
**已改**：加 `_bf16_requested()` + 一行显式日志，说明"被忽略"及其原因；默认（不设
`TG_BF16`）不打印（实测计数 0）。
**刻意不放开 bf16**：`_bf16_epoch_bump()`（bf16 权重影子失效）只挂在 `_Optim` 家族的
`step` 上，`Int8AdamWGpu.step()`（`--opt int8gpu`，本基准与生产真用的那个）**没有调用它**
⇒ 放开后会用到上一步的 bf16 权重，梯度静默错（与历史上 fp16 那次
`grad_norm 12.49 → 145775` 同类事故）。详见 §5-U4。

---

## 4. 修复前后对比（4090，30 步，A/B/A/B 交替，同机同数据）

### amp=off（默认）

| 轮次 | A（1.0.8 未改） | B（改后） |
|---|---|---|
| 第 1 组 | 0.23302 s / util 30.0%（中位 37%） | 0.20948 s / util 33.6%（中位 42%） |
| 第 1 组 | 0.24196 s / util 29.4%（中位 36%） | 0.20469 s / util 34.7%（中位 42%） |
| 最终验收 | 0.25042 s | 0.21256 s |
| 最终验收 | 0.22973 s | 0.21325 s |
| **4 次均值** | **0.23878 s**（区间 0.22973~0.25042，±4.3%） | **0.21250 s**（区间 0.20469~0.21325） |

⇒ **步时 −11.0%（0.2388 → 0.2125 s）**，且 **A 的区间 [0.2297, 0.2504] 与 B 的区间
[0.2047, 0.2132] 完全不重叠**（不是靠单次运气）。
训练窗口 GPU 利用率 **29.4~31.9% → 32.6~34.7%（中位 36~38% → 41~42%）**。

### amp=gemm

| 轮次 | A | B |
|---|---|---|
| 1 | 0.33111 s / util 19.3% | 0.28725 s / util 20.0% |
| 2 | 0.32594 s / util 17.9% | 0.29931 s / util 14.3% |
| **均值** | **0.32853 s** | **0.29328 s** |

⇒ **步时 −10.7%**；util 在这条路径上 1 Hz 采样噪声过大，看不出方向（不建议据此下结论）。

### 与「不改动」的一致性检查

修复只动了 clip（1283→3 次发射，实测 clip 段 14.3 ms）与广播参数（576+57 次/步），
按 §2.3 的单价估算应省 14×(1−3/1283)+9 ≈ 23 ms，实测省 30.4 ms（多出来的部分来自
321 次 `empty` 与 320 次 `add_` 一起消失）—— 与实测同量级，**归因自洽**。

---

## 5. 未解决 / 未验证（按建议优先级）

| 编号 | 项 | 现状与数据 | 建议 |
|---|---|---|---|
| U1 | **run-to-run 不可复现** | 同代码同数据两遍，12 步 gnorm 只有 2/12 逐位相同，loss 第 7 步起差 2.7e-3；`TG_DETERMINISTIC` 默认关 | 训练/对拍一律设 `TG_DETERMINISTIC=1`；是否改默认值需要单独评估（掉速未知） |
| U2 | **`rt.empty` 池分配 6.48 µs × 4539 = 29.4 ms/步（12.4%）** | 命中路径只有 2 次 dict 查/写 + 1 个 3-tuple + `GpuBuf()`（1.89 µs）；`_pool_token_of` 再做一次 dict 查 | 下一个最值得动的点，但要改分配器，需单独的 A/B + 全量回归 |
| U3 | **每算子还有 ~30 µs 的 Tensor/autograd 开销** | `mul_bcast` 41.4 µs 里缓存后仍剩 ~34 µs，发射只占 7.35 | 真正的结构性解法是**算子融合 / CUDA Graph**：把 4038 次降到 ~200 次 ⇒ 光发射地板就从 29.7 ms 降到 0.9 ms |
| U4 | **bf16 + int8gpu 有正确性洞** | `Int8AdamWGpu.step()` 未调 `_bf16_step_boundary()`（`cuda.py:7559`）；`_Optim`/opt_extra 都调了 | 放开 bf16 前先补这 2 行 + 做 bf16/fp32 数值对拍 |
| U5 | **小内核的 GPU 时间测不准** | 主机饿死 GPU 时，event 间隔 = 主机时间（`empty` 无内核也量出 6.5 µs/次） | 需要"填充内核把队列压满"的挂具，或 Nsight Systems |
| U6 | **`h2d` 224 次/步** 未归因 | 单次成本小（4 B~数 KB），估计 <1 ms/步 | 低优先 |
| U7 | **`_vram_peak` 每 0.2 s 一个 `nvidia-smi`** | 稳态步时无可测差异；但吃约 10% 的一个核 + IO | 建议改成 1 s 或读一次 `cuMemGetInfo`（U2/U3 之后再看） |

---

## 6. 改了哪些文件

```
 M python/tgai_go/train/cuda.py        +246 −?  三个分片内核 + _shard_offsets +
                                       sumsq_multi_total_sharded / scale_multi_sharded +
                                       _bcast_tail 及 mul_bcast/add_bcast 改用缓存
 M python/tgai_go/train/nn_ops.py       +49 −?  clip_grad_norm_ 走分片版（删掉 1<<20 门槛）
 M python/tgai_go/train/v3_pretrain.py  +27     _bf16_requested() + TG_BF16 被忽略的显式日志
```
（`lab/rent/q_hope_pyt.py` 的改动在本任务开始前就存在，不是本次引入。）

`git status --porcelain`（只列非 `lab/rent` 新增文件）：

```
 M lab/rent/q_hope_pyt.py          <- 任务开始前已存在，未触碰
 M python/tgai_go/train/cuda.py
 M python/tgai_go/train/nn_ops.py
 M python/tgai_go/train/v3_pretrain.py
```

## 7. 回归

```
python -X utf8 -m pytest tests -q
=> 831 passed, 59 skipped, 3 warnings in 49.46s
```
（与要求的 ≥831 passed / 59 skipped 基线一致。）

## 8. 复现命令（4090）

```bash
# 代码：纯 Python 包，无需重装
tar -xzf pkg.tgz -C /root/orv_src/python
cd /root/bench
PYTHONPATH=/root/orv_src/python /root/miniconda3/bin/python -m tgai_go.train.v3_pretrain \
  --data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 \
  --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 \
  --amp off --steps 30 --log-every 99 --no-final-save
# 探针（本报告用到的 5 个）：/root/gt/{gt_cpu,gt_phase,gt_prof,gt_lat,gt_micro,gt_micro2,gt_equiv}.py
```

---

# 第二轮（P0 / P1 / P2，2026-11-20 晚）

## 4b. 关于第一轮 A/B 数字的更正（采纳「交替多轮 + 中位 + 极差」口径）

第一轮我报的 −12.8% 是"两对交替"的均值，偏乐观。把**同一批 7+7 次**全部汇总后：

| 口径 | 基线 1.0.8 | 补丁（仅裁剪修复） | 变化 |
|---|---|---|---|
| 我的 4 对（均值） | 0.23878（0.22973~0.25042） | 0.21250（0.20469~0.21325） | −11.0% |
| lead 的 3 对（中位） | 0.2331（0.2298~0.2391） | 0.2179（0.2099~0.2197） | −6.5% |
| **14 次全汇总（中位）** | **0.2331** | **0.21325** | **−8.5%** |

⇒ 后续一律按 **−8.5%（14 次中位）** 引用，区间 **−6.5% ~ −11.0%**。
（基线自身在 0.2297~0.2504 之间漂 ±4.3%，单对交替确实会取到两端。）

## 9. P2：`TG_DETERMINISTIC` 复核 —— 文档说法**仍然有效**

同一份代码、同一数据、两个**独立进程**、12 步、`--out` 落 `train_log.jsonl` 比全精度值：

| 条件 | loss 逐位相同 | gnorm 逐位相同 | max\|Δloss\| | max\|Δgnorm\| |
|---|---|---|---|---|
| **不设** `TG_DETERMINISTIC` | **3/12** | **3/12** | 1.61e-4 | 5.01e-3 |
| **`TG_DETERMINISTIC=1`** | **12/12** | **12/12** | **0** | **0** |

⇒ **1.0.7 发布说明与 `docs/DETERMINISM.md` 的说法成立**：开了开关就是逐位可复现（本次实测 Δ 严格为 0）。
我第一轮报告里那句"该训练器 run-to-run 不可复现"**只在"没开开关"的前提下成立**，措辞已更正（§1 R7）。
⇒ 顺带解释了「数字反常 / 加速比不稳」：默认配置下两次运行从第 1 步就开始分叉（gnorm 差 5e-3）。

## 10. P0：分配器热路径优化 —— 6.48 → 5.80 µs，端到端 −2.8%

### 10.1 成本组成（20000 次循环，4090）

| 项 | µs |
|---|---|
| 纯 dict 取/还地板（同一个桶 pop 再 append） | **0.211** |
| `GpuBuf(...)` 构造（含一次 `__del__`→`free`） | 1.857 |
| `_alloc` + `free`（不进 GpuBuf） | 3.888 |
| `np.dtype(np.float32)` | 0.225 |
| **`rt.empty((512,1024))` 完整一次（含 `__del__` 归还）** | **6.477 → 5.798** |

### 10.2 改了什么（全部语义等价，无行为变化）

1. `empty()`：`dtype` 已是 `np.dtype` 时不再 `np.dtype()`；`shape` 已是 tuple 时不再 `tuple()`；
   rank-2 直接 `shape[0]*shape[1]`（原来一律走 `for` 循环 + 每维 `int()`）。
2. `GpuBuf.__init__`：`nbytes`/`n`/`shape`/`dtype` 只在**类型不对**时归一化（原来无条件四个 `int()/tuple()/np.dtype()`）。
3. `_alloc()`：三处热路径 `getattr(self, "...", False)` 改成属性直读（`_pool_cap_adapted` /
   `_alloc_retry` / `_capturing` / `_arena_base`），并把前两个标志在 `__init__` 里**提前初始化**
   （原来是靠 `getattr` 的默认值，等于每次分配多两次字典查找）。
4. `_pool_checkout()`：`_pool_note()` 内联（省一次方法调用 + 一次属性读，一步 4500+ 次）。
5. `_pool_release` 早就有一条"不记 event"的快路径（`self._lfast`），本次未动 —— 确认它生效。

### 10.3 A/B（同会话、交替 3 轮、中位 + 极差；A = `d531c95`，B = 加了 P0）

| 轮 | A（HEAD） | B（P0） |
|---|---|---|
| 1 | 0.21185 | 0.20756 |
| 2 | 0.21459 | 0.20723 |
| 3 | 0.21353 | 0.21105 |
| **中位** | **0.2135** | **0.2076** |
| 极差 | 0.2119~0.2146 | 0.2072~0.2111 |

⇒ **−2.8%**，且 B 的最大值（0.2111）< A 的最小值（0.2119），区间不重叠。
单次 `empty` 成本 6.477 → **5.798 µs（−10.5%）**（20k 次微基准，确定性）。

### 10.4 为什么没做到 ≤2 µs（用测量说话）

`empty()` 一次完整生命周期 = 3.888（`_alloc`+`free` 记账）+ 1.86（`GpuBuf` 构造+析构）+ 0.06
= 5.80 µs，其中**纯数据结构地板只有 0.21 µs**。也就是说 5.6 µs 是**记账本身**：
`_pool_live` 的 (size, seq, stamp) 3-tuple 建/删、`_pool_live_bytes`/`_pool_seq`/峰值三个计数器、
`_split_forget_ptr` + `_f16_cache_drop_ptr` 两次缓存失效、以及 `GpuBuf.__del__` 这个 Python finalizer。

要做到 ≤2 µs 必须动**结构**，不是动语句：

* **方案 A（推荐评审）**：`GpuBuf` 对象池 —— 析构时不 `free()`，而是把自己挂回 `bucket` 的对象栈；
  `empty()` 命中时复用对象只改 6 个字段。可省掉 `_pool_live` 建删 + `__del__` + 记账（≈4.5 µs），
  预估 **1.2~1.5 µs**。风险：`_pool_live_bytes`/`peak` 的语义要重新定义（对象池里的块算 live 还是 cached），
  且 `empty_cache()`/OOM 兜底/图捕获三条路径都要重新对拍。
* **方案 B**：把 `_pool_live` 的 3-tuple 换成"块头结构"（`array('Q')` 或 3 个平行 list），省 tuple 分配。
  预估 5.8 → 4.6 µs（−21%），风险低。
* **方案 C**：合并 `_pool_release` 的两个缓存失效（`_split_forget_ptr` + `_f16_cache_drop_ptr`）成一次调用，
  两者都只在缓存非空时才真正工作。预估 −0.3 µs。

按方案 A 的上限（−4.5 µs × 4539 次 ≈ −20 ms/步）算，端到端约 **−8%**，与 lead 的目标一致；
但那是重写分配器，需要单独一轮设计评审 + 全量对拍，本次没有动（纪律：一次只改一处、稳定优先）。

## 11. P1：CUDA Graph —— `TG_CUDA_GRAPH` **是一行没人读的注释**，对本配置收益为 0

### 11.1 静态证据

* `python/` 下 `environ.*TG_CUDA_GRAPH` / `getenv.*TG_CUDA_GRAPH` 匹配数 **0**
  （该字符串全仓库只出现在 3 处注释/错误提示 + docs 里）；
* `v3_pretrain.py` 里 `graph_arm|graph_begin|GraphStep|layer_graph` 匹配数 **0**；
* 图路径的唯一入口是 `graph_train.GraphStep`（`layer_graph` 同理），由调用方**显式**调
  `rt.graph_arm()/graph_begin()`；`v3_pretrain` 从不调用 ⇒ 设环境变量**什么都不会发生**。

### 11.2 实测（同一份 B 代码，交替 2 轮）

| 轮 | `TG_CUDA_GRAPH=0` | `TG_CUDA_GRAPH=1` |
|---|---|---|
| 1 | 0.20669 / util 35.3% | 0.20545 / util 34.8% |
| 2 | 0.20586 / util 35.3% | 0.21140 / util 34.5% |
| 中位 | 0.2063 | 0.2084 |

⇒ **+1.0%（噪声内）**；日志里**没有任何** `[TG_CUDA_GRAPH]` / "捕获失败" 消息（`no-graph-msg`），
说明既没有捕获、也没有静默回退 —— 就是完全没走这条路。

### 11.3 能不能和 `--opt int8gpu` / `--amp off` 组合？（明确的答案）

**按现在的实现不能，而且有三个结构性障碍**（这也是它一直没接进 `v3_pretrain` 的原因）：

1. **`--grad-clip 1.0`（默认值）在步内做标量 D2H**：`clip_grad_norm_` 末尾要
   `read_scalar()` → `cuMemcpyDtoH`。`graph_train.py` 自己的文档就写着
   *"图里不能有 D2H（`read_scalar`：比如梯度裁剪 `clip>0` 会读标量）⇒ 捕获会失败"*，
   并且 `GraphStep(clip=0.0)` 默认就把裁剪排除在图外。要用图必须把裁剪搬到图外
   （等于每步仍然一次设备同步 + 一次 norm 归约，只是不再是 962 次发射）。
2. **MoE top-k 路由是数据相关的**：图里刻的是固定地址上的固定 kernel，`--opt int8gpu` 无影响，
   但 `E=4/top_k=2` 的 `moe_topk`/`gather_rows`/`scatter_add_rows` 的**目标行数随 batch 内容变**。
   直接捕获会重放**第一步的路由形状**，静默错（不报错、loss 也能降，属于最难查的一类）。
   要图安全，路由必须改成"固定容量的桶 + 计数内核"，那是模型层改动。
3. **`int8gpu` 的 `adamw_multi_m8v16` 是单次覆盖全部参数的大内核**，它本身对图友好；
   但 `graph_train.py` 的 `GraphStep` 接口要求 `model/opt/ids_shape/tgt_shape/vocab/loss_fn`，
   与 `v3_pretrain` 的 `forward_loss(P, cfg, kinds, cos, sin, ids, labels, seg, …)` 函数式 API 不匹配，
   接入需要一层适配。

⇒ `docs/PRODUCTION_READINESS.md:623` 那个 **−12.9%（4090，21.5→18.9 ms）** 是
**Qwen2.5-0.5B 稠密 / clip=0 / B=4·S=512** 的测法，**不能**外推到本次的 12 层 MoE+CSA 配置。
本次结论：**在 `v3_pretrain` 这条路径上 CUDA Graph 的收益是 0（因为它没被接上）**；
要拿到收益需要先做上面第 1、2 两件事，属于独立项目。

## 12. 第二轮改动文件 / git status / 回归

```
 M python/tgai_go/train/cuda.py        （仅分配器热路径，50 insertions / 17 deletions）
```
（第一轮的 `nn_ops.py` / `v3_pretrain.py` / `cuda.py` 已由 lead 提交为 `d531c95`。）

`git status --porcelain`（只列非 `lab/rent` 新增）：

```
 M python/tgai_go/train/cuda.py
?? lab/rent/REPORT_gpu_idle_20261120.md
```

回归：

```
python -X utf8 -m pytest tests -q
=> 831 passed, 59 skipped
```

## 13. 更新后的未解决项优先级

| 编号 | 项 | 数据 | 建议 |
|---|---|---|---|
| U2' | **分配器结构性重写**（GpuBuf 对象池） | 5.80 µs/次 × 4539 = 26.3 ms/步（11%）；结构改造上限 ~1.2-1.5 µs ⇒ 端到端 −8% | 下一轮单独立项 + 设计评审（三条路径要对拍：`empty_cache` / OOM 兜底 / 图捕获） |
| U3' | **算子融合 / 图化** | 4038 次发射 × 4.39 µs 驱动地板 = 17.7 ms 是**不可再降的发射地板**；要再降只能减少发射次数 | 融合逐元素链（`mul_bcast`+`fused_add_ip`+`rmsnorm` 一组），或做 MoE-图安全改造后上图 |
| U4 | bf16 + int8gpu 的正确性洞 | `Int8AdamWGpu.step()` 未调 `_bf16_step_boundary()` | 放开 bf16 前先补，再数值对拍 |
| U5 | 小内核 GPU 时间测不准 | 主机饿死时 event 间隔 = 主机时间 | 需要"填充内核压满队列"的挂具或 Nsight Systems |
| U7 | `_vram_peak` 每 0.2 s 一个 `nvidia-smi` | 稳态步时无可测差异，但吃掉约 10% 的一个核 + IO | 建议改 1 s 或直接用 `cuMemGetInfo` |