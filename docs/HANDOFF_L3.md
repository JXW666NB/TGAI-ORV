# L3 闭环交接清单（③-1 长稳 / ③-2 双卡 / ③-3 bf16）

> 用途：**当前没有可用硬件**，③ 卡在三项上。这份清单让你（或任何人）租到机器后
> 照着做就能把 ③ 判完，不需要重新摸索、也不会误判。
> 纪律：**不许用 V100 单卡推算双卡数字**；不许为了"好看"放松判据。

---

## 0. 需要什么样的机器

| 用途 | 最低要求 | 说明 |
| --- | --- | --- |
| ③-1 长稳复验 | **1 张** sm_80+ 卡（A10/A100/4090 都可），24 GB 显存，24 小时独占 | 当前代码要在**健康机器**上跑满 24h |
| ③-2 双卡加速比 | **2 张同型号卡在同一台机器**，支持 P2P | 不同机不行（用 IPC 句柄）|
| ③-3 bf16 对决 | sm_80+（有 bf16 Tensor Core）| **V100(sm_70) 物理不可能**，必须换卡 |

> 我们自己的 V100 有**已知缺陷**（功率高 → CUDA 上下文崩，`rc=700`，PyTorch 同样中招），
> 不能用来做长时间验证。证据见 `docs/PRODUCTION_READINESS.md`「长稳复验结论」。

到机先验：

```bash
nvidia-smi -L                      # 卡数与型号
nvidia-smi --query-gpu=driver_version --format=csv
nvidia-smi topo -m                 # 双卡时看 P2P/NVLink
python tools/am_preflight.py       # 环境/GEMM/训练一步 体检
```

---

## 1. ③-1 长稳复验（约 24 小时，无人值守）

```bash
# 建议放后台/定时任务，日志落盘
python -X utf8 tools/soak_test.py --minutes 1440 --batch 64 \
    --sample-steps 200 --csv soak_24h.csv > soak_24h.log 2>&1
```

**看什么**（每 200 步一行）：

| 指标 | 通过要求 |
| --- | --- |
| `loss` | 稳定在 0.08~0.10 区间波动，**无 NaN/inf、无单调上升** |
| `peak` / `live` | 不随时间增长（本项目实测 2.525 GiB / 1.332 GiB 恒定）|
| `free` | 不持续下降 |
| `rss` | 不持续增长（无主机内存泄漏）|
| 进程存活 | 跑满 1440 分钟，`EXITCODE=0` |

**判定**：以上全满足 ⇒ ③-1（当前代码）通过。把它写进
`docs/PRODUCTION_READINESS.md` 收官摘要的 ③-1 行，并注明**机器型号 + 代码 commit**。

---

## 2. ③-2 双卡加速比（约 15 分钟）

```bash
# 2.1 协议自检（单卡也能跑，必须先过）
python tools/dp_speed.py --check-only
#     判据：全归约相对误差 ≤1e-6（历史实测单机双进程 3.98e-08）

# 2.2 单卡基线
python tools/dp_speed.py --world 1 --steps 10 --warmup 3 \
    --layers 24 --global-batch 8 --seq 512

# 2.3 双卡实测
python tools/dp_speed.py --world 2 --steps 10 --warmup 3 \
    --layers 24 --global-batch 8 --seq 512
```

**判据**：`throughput(2) / throughput(1) ≥ 1.7`。

| 结果 | 含义 / 下一步 |
| --- | --- |
| ≥1.7 | ③-2 通过，记录数字 |
| 1.3~1.7 | 先看 `nvidia-smi topo -m` 是否走 PCIe（无 NVLink）；再试更大 global-batch（16/32），每卡 batch 太小会拉低 GEMM 效率 |
| <1.3 | 多半是全归约成瓶颈：先用 `--layers 24` 加大计算量；仍不行则**如实记录实测值 + 拓扑**，不改判据 |

⚠ **必须**保持 `--global-batch` 固定（脚本默认如此），否则加速比是把 batch 放大换来的假收益。

---

## 3. ③-3 bf16 正面对决（sm_80+ 才能做）

现有实现是**激活存 fp16**（`cfg["MLP_FP16"]` + `amp_mlp.GradScaler`），V100 上实测
比 PyTorch bf16/fp16 AMP 慢 **1.15~1.21×**（209~221 vs 182.9 ms）。
要在 bf16 上对决需要：

1. 把 AMP 的存储类型从 fp16 扩到 bf16（`cuda.py` 已有 `cast_f32_to_bf16` / `cast_bf16_to_f32`，
   检查点层 `dtypes.py` 也已支持 bf16）；
2. bf16 GEMM（`cublasGemmEx` + `CUDA_R_16BF`），**需要 sm_80+**；
3. 重新跑同口径对照（`python tools/dp_speed.py` 不适用，用步骤基准：见
   `docs/BENCHMARK_RIGOR.md` 的口径纪律）。

> bf16 的好处：指数范围与 fp32 相同 ⇒ **理论上不需要 GradScaler**、不会溢出尖峰。
> 这是 bf16 相对 fp16 的核心优势，值得在 sm_80+ 上验证。

---

## 4. 记录模板（照抄填空）

```markdown
### ③ 复验记录（<日期>，<机器型号> ×<卡数>）

| 项 | 结果 | 证据 |
| --- | --- | --- |
| ③-1 长稳 24h | <通过/未通过> | <日志路径>，末段 loss=…，peak=… |
| ③-2 双卡加速比 | <x.xx×> | world=1 <ms/step>，world=2 <ms/step>，global-batch=8 |
| ③-3 ACT-fp16 vs torch AMP | <x.xx×> | 我们 <ms> vs torch <ms>，同口径 N 步 |
| 代码 commit | <sha> | `git rev-parse HEAD` |
```

---

## 5. 反模式（别做）

* ❌ 用 V100 单卡数据推算双卡加速比
* ❌ 把 `--global-batch` 跟着 world 一起放大（假加速）
* ❌ 计时循环里读设备标量（会虚增步时，见 `docs/BENCHMARK_RIGOR.md`）
* ❌ 长稳没跑满就写"通过"（写"进行中 h/24"）
* ❌ 在跑 soak 的同时跑别的 GPU 任务（互相污染，2026-10-19 已踩过）