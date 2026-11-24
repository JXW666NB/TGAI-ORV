# 数据并行「反向即归约」（DP gradient overlap）— 实现与实测留档

日期：2026-11-21
机器：2×RTX 5090 31.4 GiB（实测 `p2p=False`，拓扑 `SYS`），单机 2 rank
模型：V3 真尺寸 12 层 / E=4 / d_ff=3328 / d=1024 / n_q=16 / vocab=96000 / `--csa 4` / 848.645M 参数
梯度：848.6M × fp32 = **3.16 GiB**
数据：`/root/bench/data2048_v96000`，每 rank batch 1

## 1. 改了什么

| 文件 | 内容 |
| --- | --- |
| `train/dp_overlap.py`（新） | 桶布局 `build_plan` / `plan_order_adapter`；`OverlapSession`（pending 记账、硬校验、stats）；三种传输：`NumpyTransport`（CPU 逐位对拍用）、`NcclBucketTransport`（第二非阻塞流 + event 同步）、`HostBucketTransport`（后台线程 + 双流 + 分批 barrier） |
| `train/__init__.py` | `_BWD_OVERLAP` 钩子 + `_set_bwd_overlap()`；`Tensor.backward` CPU/CUDA 两条路径在**叶子 `.grad` 写完**处调 `leaf_ready`；`T.checkpoint` 打 `bw._is_checkpoint` 标记 |
| `train/dist.py` | `DataParallel.begin_overlap/finish_overlap/abort_overlap`；重叠专用非阻塞流；`stats()` 暴露 `overlap`；NCCL 通道数默认 8（同机 PCIe 实测 134→86 ms） |
| `train/dist_nccl.py` | `all_reduce_group(entries, stream=...)`（可选流）；ZeRO 用的可选符号（与本任务无关，但同文件） |
| `train/cuda.py` | `launch/d2h_into/h2d_into/copy_d2d/scale_inplace/add_` 的可选 `stream=` 参数；`event_create/record/sync/elapsed/destroy` |
| `train/v3_pretrain.py` | 训练循环接线：反向前 `begin_overlap`、反向异常 `abort_overlap`、反向结束 `finish_overlap`（退回 `all_reduce_grads`）；`--tie-embeddings`；`[dp-ov]` stats 日志；顺手修了 `--amp` help 里未转义 `%` 导致 `--help` 崩溃 |
| `train/v3_train_cli.py` | `_checkpoint_layer` 的反向打 `_is_checkpoint`（重叠看到就整步退回不重叠） |
| `tests/test_dp_overlap.py`（新） | 22 个纯 CPU 测试：桶布局、tied 逐位、pending 不归零必须 raise、梯度累积/检查点退回、2-rank 平均逐位、stats 字段 |

## 2. 三条铁律的实现方式

1. **tied 不会归约半成品**：钩子挂在 `Tensor.backward` 里**写 `leaf.grad` 的那个点**
   （`__init__.py:866/1413`）。ORV 的 `_acc` 只把各路径贡献累加进中转缓冲，
   `leaf.grad` 每次反传只写一次（拓扑序保证所有贡献已累加）；tied 的同一张量也只在
   拓扑序里出现一次 ⇒ `pending=1`。CPU 上用 `tie_embedding=True` 与不重叠版做
   `assert_array_equal` 逐位对比（`test_tied_embedding_bitwise_identical_to_no_overlap`）。
2. **pending 不许静默漏**：`finish()` 逐参数核对；"该到而没到、但 `.grad` 已不是
   None" ⇒ **抛 RuntimeError**（`test_missing_hook_is_loud` 守着）。确实没梯度的
   （pending 0 / soft）按不重叠路径语义补 0 梯度，计入 `stats()["zero_filled"]`。
3. **集合通信序列各 rank 逐字相同**：桶布局只由 `params` 静态决定（各 rank 相同），
   提交严格 FIFO；**绝不**按"谁先就绪"动态组桶（MoE 路由数据相关）。

## 3. 实测（全部在这台机器上跑过）

### 3.1 正确性（`adamw`，2 步，2 rank，逐位 = `==` / `assert_array_equal`）

| 对拍 | 结果 |
| --- | --- |
| world=1：`TG_DP_OVERLAP=0` vs `1`（untied） | **逐位相同** |
| world=1：`TG_DP_OVERLAP=0` vs `1`（tied） | **逐位相同** |
| world=2 `nccl`：重叠 on vs off（untied） | **逐位相同**，且 rank0/rank1 的 gnorm 逐位相同 |
| world=2 `host`：重叠 on vs off（untied） | **逐位相同**，rank0/rank1 gnorm 逐位相同 |
| world=2 `nccl`：重叠 on vs off（tied） | **逐位相同**，rank0/rank1 gnorm 逐位相同 |
| world=2 `host`：重叠 on vs off（tied） | **逐位相同**，rank0/rank1 gnorm 逐位相同 |

> 注意：带 tied 的那 4 条是在 **host 传输改写成双流流水线之前** 的代码上验证的；
> 双流版 host 只做了 CPU 单测，GPU 未验证（机器在验证前被回收，见下）。

### 3.2 性能（交替 3 轮 × 20 步，median[range]，`--opt int8gpu`，`amp off`）

配置：NCCL、`TG_DP_BUCKET_MB=256`、`TG_DP_OVERLAP_SLOTS=2`、NCCL 8 通道（`dist._init_nccl` 默认）。

| seq | 单卡 step | 2-rank step | 吞吐比（2×T1/T2） | 归约忙碌 | 显存峰值（单/双） |
| --- | --- | --- | --- | --- | --- |
| 512 | 0.1406 / 0.1364 / 0.1486 s | 0.2565 / 0.2561 / 0.2597 s | **1.096×**（范围 1.065–1.144） | 111.4–111.7 ms/步 | 15.5 / 16.8 GiB |
| 2048 | 0.2515 / 0.2452 / 0.2542 s | 0.3998 / 0.4285 / 0.4058 s | **1.253×**（范围 1.144–1.258） | 111.8–112.0 ms/步 | 31.5 / 31.7 GiB |

对照（基线，同一台机、同配置、1.0.15 release 的**不重叠**路径）：
seq512 2-rank **0.84×（nccl）/ 0.87×（host）**；seq2048 **1.20×/1.21×**。
⇒ 重叠把 seq512 从 0.84× 提到 1.10×、seq2048 从 1.20× 提到 1.25×，**但没达到验收线**。

### 3.3 归约本身的物理成本（裸测，`lab/rent/_ov_arlat.py`）

3.16 GiB fp32 all-reduce、2 rank、无计算并发：

| NCCL 配置 | 时间 | 等效带宽 |
| --- | --- | --- |
| 默认（`NCCL_MIN/MAX_NCHANNELS` 未设） | 133.9 ms | 25.4 GB/s |
| `NCCL_PROTO=LL` | 543.9 ms | 6.2 GB/s |
| `NCCL_PROTO=Simple` / `NCCL_ALGO=Ring` | 134.2 / 134.5 ms | 25.3 GB/s |
| `NCCL_MIN=MAX_NCHANNELS=4` | 102.0 ms | 33.3 GB/s |
| **`NCCL_MIN=MAX_NCHANNELS=8`** | **86.5 ms** | **39.2 GB/s** |
| `NCCL_MIN=MAX_NCHANNELS=16` / `32` | 88.0 / 93.8 ms | 38.1 / 35.9 GB/s |
| `NCCL_SHM_DISABLE=1`（禁同机 shm） | 1335 ms | 2.5 GB/s |

⇒ 8 通道是同机 PCIe 的实测最优点，已设为 `dist.py::_init_nccl()` 的默认
（`os.environ.setdefault`，用户显式设置可覆盖）。归约本身的物理下限 ≈ 86 ms
（PCIe 5.0 x16 双方向实测约 39 GB/s/方向）。

### 3.4 为什么没达标（关键实测归因）

| 事实 | 数字 |
| --- | --- |
| ORV 单卡一步 = fwd 54 ms + bwd 100 ms + opt 15 ms + 数据/其它 145 ms（seq512, host 不重叠路径） | 0.314 s |
| 用 **host DMA 路径**（拷贝引擎与计算真并发）时 bwd **不膨胀** | 70 ms（≈ 纯反向时间） |
| 用 **NCCL 路径**时 bwd 被拖慢 | 70 → 183 ms（+113 ms ≈ 归约忙碌时间） |
| 即 NCCL 的 all-reduce 内核与 ORV 的 GEMM/elewise 在这台 PCIe-only 的 5090 上**基本不共驻** → 归约时间几乎 100% 叠加到反向里 | |

于是物理上限是：`step_dp ≈ step_single + (comm − bwd)`（comm 无法与 bwd 并发时）。
seq512：0.143 + (0.112 − 0.070) ≈ 0.185 s（理想并发）→ 上限 ≈ 1.55×；
实测 1.10× 说明 NCCL 内核与计算**几乎完全串行**。若换成 PyTorch DDP 的同款对照
（本机实测 bf16 梯度）：其 bwd 也由 81 → 131 ms（+50 ms，归约 70 ms），
即同样有 ~70% 串行化 —— 这不是 ORV 独有的问题。

结论：**在「3.16 GiB fp32 梯度 + 无 P2P」这两个物理条件下，seq512 ≥1.35× 达不到**；
seq2048 实测 1.25× 也低于 1.5×。若要把吞吐比做上去，只有三条路：
① 梯度通信降到 bf16/fp16（会破坏「与不重叠版逐位相同」的判据，且改变优化语义）；
② 换有 P2P/NVLink 的机器（通信时间从 86 ms 降到 ~10 ms 量级）；
③ 把归约真正藏进反向（需要让 NCCL 内核与计算共驻，或改用能走拷贝引擎的归约实现；
本仓库的 host 双流流水线就是 ③ 的一次尝试，见下）。

## 4. 未验证 / 风险

- **host 双流 + 分批 barrier 版**（`HostBucketTransport`，默认 `TG_DP_HOST_BATCH=slots`）：
  只过了 22 个 CPU 单测与 `py_compile`；**GPU 未验证**（改写完成后机器就被回收）。
  已验证的 host 数字来自之前的单流串行版：bwd 不膨胀（70 ms），但归约暴露 ≈156–186 ms。
- **tied + 双流 host** 未在 GPU 上对拍。
- `NCCL_MIN/MAX_NCHANNELS=8` 的默认值只在「同机 PCIe」验证过；多机/NVLink 拓扑未验证。
- 显存：cv 判据是「峰值涨幅 ≤1.0 GiB」。实测 seq512 +1.0~1.3 GiB（768 MB 桶缓冲 +
  NCCL/池开销），seq2048 +0.2~0.3 GiB。**seq512 略有超标**。
- `pending` 判据、`TG_DP_OVERLAP=0` 退回、`TG_DETERMINISTIC`、无 atomicAdd：
  代码层面保证（钩子只读 `.grad`；不含 atomicAdd）；确定性开关未额外回归。

## 5. 复现方式

```bash
# 裸 NCCL 吞吐（2 rank）
bash lab/rent/_ov_arlat.sh

# 正确性对拍（world1/2、tied、host/nccl；需要 2 卡）
bash lab/rent/_ov_correct_final.sh

# 交替 3 轮 × 20 步性能（seq512 / 2048）
bash lab/rent/_ov_final.sh 512
bash lab/rent/_ov_final.sh 2048
python lab/rent/_ov_final_sum.py 512
```

CPU 单测：`python -X utf8 -m pytest tests -q`（**1033 passed / 64 skipped**，含本任务 22 个）。

## 6. 4090 复测（2026-11-21 新机，2×RTX 4090 24 GiB）

新机开着 2×4090（PCIe 4.0，同样 p2p=False），同一份代码原样复测：

| 项 | 实测 |
| --- | --- |
| 裸 NCCL all-reduce 3.16 GiB（默认通道） | **448.2 ms**（7.6 GB/s） |
| 裸 NCCL all-reduce（MIN/MAX_NCHANNELS=8） | **353.6 ms**（9.6 GB/s） |
| 单卡 seq512 848.6M（20 步） | 0.2311 / 0.2338 / 0.2311 s（3 轮） |
| 2-rank seq512 不重叠（3 轮） | 0.6902 / 0.6756 / 0.6661 s |
| 2-rank seq512 重叠（3 轮） | 0.6570 / 0.6570 / 0.6599 s |
| **seq512 吞吐比（不重叠 / 重叠）** | **0.684× / 0.704×**（3 轮中位） |
| 2-rank seq512 峰值显存（单 / 双） | 15.4 / 16.7 GiB |
| seq2048 单卡 | 可跑：0.849 s/步（2 步），峰值 **24073 MiB / 24564 MiB** |
| **seq2048 2-rank** | **跑不了：CudaError: 显存不足（申请 0.03 GiB，设备 23.55 GiB 已满）** |

4090 结论（与 5090 的差异来自物理）：

* 4090 上 comm 是 353 ms，而单卡整步只有 231 ms ⇒ **3.16 GiB fp32 梯度的 2-rank DP
  本身就是净亏**（0.68×），重叠最多把暴露的尾巴削掉一点（→0.70×）。
  「seq512 ≥1.25×」在这台机器上**物理上不可达**：即使通信 100% 藏进反向，
  上限也只有 2×0.231/(0.231+0.354-0.10) ≈ 0.95×。
* seq2048 在 24 GiB 卡上单卡就贴顶（剩 491 MiB），2-rank 再多 768 MB 桶缓冲直接 OOM，
  所以 seq2048 的判据**在这台机器上无法测**（不是未达标，是跑不起来）。

正确性（4090，同一套对拍脚本）：

| 对拍 | 结果 |
| --- | --- |
| world=2 nccl / host：重叠 on vs off，untied | **逐位相同**，rank0/rank1 gnorm 逐位相同 |
| world=2 nccl：重叠 on vs off，**tied** | **逐位相同**，rank0/rank1 gnorm 逐位相同 |
| world=1：重叠 on vs off | gnorm 在第 2 步差 **1.9e-7**；同一配置连跑两次也可能差（MoE 反向原子加顺序的运行间非确定性），与重叠代码无关（world=1 时重叠是纯空操作） |

结论不变：**在这两种「无 P2P 的 PCIe 机器 + 3.16 GiB fp32 梯度」上，反向重叠
能改善（5090：0.84→1.10×；4090：0.68→0.70×），但达不到 1.25–1.5× 的验收线。**
要让 DP 变成正收益，必须先降通信字节（bf16 梯度）或换有 P2P/NVLink 的机器。
