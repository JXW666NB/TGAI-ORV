# 双卡加速比执行手册（③-2）

> 目标判据：**2 卡 vs 1 卡 ≥ 1.7×**（同一全局 batch、同一模型、同口径）。
> 这份手册是给「机器到手后 30 分钟内出结论」用的，不需要重新摸索。

## 0. 前提

| 项 | 要求 | 怎么确认 |
| --- | --- | --- |
| 2 张卡在同一台机器 | 同机才用 P2P IPC | `nvidia-smi -L` 出现两张 |
| 支持 P2P 访问 | HBM 卡一般可以 | `nvidia-smi topo -m` 看 NVLink/PCIe |
| 驱动 + CUDA ≥ 12 | 引擎走 driver API | `nvidia-smi --query-gpu=driver_version --format=csv` |
| 主机内存 ≥ 16 GB | 建 0.5B 模型要 ~6 GB | 任务管理器 |

## 1. 体检（1 分钟）

```bash
python tools/am_preflight.py            # 环境/accept/GEMM 对照/训练一步
```

## 2. 协议自检（2 分钟，**单卡也能跑**）

```bash
python tools/dp_speed.py --check-only    # = tools/dp_selftest.py
```

判据：全归约相对误差 **≤1e-6**（历史实测单机双进程 **3.98e-08**）。
不过就不要往下走 —— 说明 IPC/句柄/槽位有问题。

## 3. 单卡基线（5 分钟）

```bash
python tools/dp_speed.py --world 1 --steps 10 --warmup 3 \
    --layers 24 --global-batch 8 --seq 512
```

记录 `RESULT world=1 sec_per_step=... throughput=...`
（V100 上用 `--layers 8` 实测是 **289.4 ms/step、27.65 样本/s**，可作对照量级）

## 4. 双卡实测（5 分钟）

```bash
python tools/dp_speed.py --world 2 --steps 10 --warmup 3 \
    --layers 24 --global-batch 8 --seq 512
```

脚本会自动给两个 rank 设 `CUDA_VISIBLE_DEVICES=0/1`，
并固定**全局 batch**（每卡 4）→ 所以加速比是**吞吐之比**，不是把 batch 放大换来的假加速。

## 5. 判定与记录

| 结果 | 结论 |
| --- | --- |
| `throughput(2) / throughput(1) ≥ 1.7` | ③-2 **通过** |
| `1.3 ~ 1.7` | 未达标：先看是否被 PCIe 带宽卡住（`nvidia-smi topo -m`），再看每卡 batch 是否太小（global-batch 8 → 每卡 4，GEMM 效率低；可试 16/32） |
| `< 1.3` | 大概率全归约成了瓶颈：用 `--layers 24` 加大计算量再测；仍不行则记录实测值 + 拓扑 |

把最终数字写进 `docs/PRODUCTION_READINESS.md` 的「收官摘要」表格（③-2 那一行），
并同步 CHANGELOG。**不许用 V100 单卡推算双卡数字。**

## 6. 已知坑

* **必须**用 `--global-batch` 固定总 batch；否则"加速比"没有意义（口径纪律见
  `docs/BENCHMARK_RIGOR.md`）；
* 计时循环内不许读设备标量（脚本已遵守：只在末尾 `sync` 一次）；
* 双卡时每卡只看得见一张卡（脚本用 `CUDA_VISIBLE_DEVICES` 隔离），
  不要在同一个进程里同时初始化两张卡 —— IPC 句柄交换会混乱；
* 跑之前确认没有别的 GPU 任务（包括我们自己的 soak）。