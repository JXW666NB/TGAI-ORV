
---

## 附录 A：`T.DUO()` —— 一行接管整机（1.0.16+）

**核心思想：不管你的训练程序怎么写，加上 `T.DUO()`，引擎自己分配 —— 专心坐轮椅。**

用户不需要知道 `init_process_group` / `local_rank` / `DistributedSampler` / `torchrun` /
`CUDA_VISIBLE_DEVICES`。ORV 自己探测拓扑与显存，自己选方案，自己起 worker、自己盯进程。

```python
import tgai_go.train as T

duo = T.DUO()                          # 全自动：探测 → 选方案 → 起 worker
print(duo.summary())                   # 它会说清"选了什么、否决了什么、为什么"
r = duo.run(["python", "train.py"])    # 或 duo.run(sys.argv)
```

**默认无脑，但旋钮全在**（给"长脑子"的用户）：

```python
T.DUO(dry_run=True)                            # 只看计划，不碰任何进程
T.DUO(devices=[0, 2, 3], strategy="data_parallel")
T.DUO(experts={0: 6, 1: 2})                    # 专家并行：0 号卡 6 专家、1 号卡 2 专家
T.DUO(n_params=3e9, seq=512, optim="int8gpu")  # 给了参数量才能算显存、判断装不装得下
```

### 它替你处理四件必踩的事

| 墙 | 引擎怎么处理 |
| --- | --- |
| 碎片 / 僵尸 / OOM | 每个 worker 在自己的进程组；`atexit` + 信号 + 异常都走幂等 `stop()`；**任何一个 rank 非 0 退出或日志命中 OOM 关键字，立刻杀掉其余**；**计划不可行时在起进程之前就拒绝** |
| 数据切分 | 设 `TGAI_DP_WORLD/RANK/JOB`；`duo.shard_of(n)` 给本 rank 的分片（各段互不重叠、并起来是全集）|
| 木桶效应 | 异构卡（V100+4090）**只能检测并警告**：同步 DP 一定被最慢的卡拖住，这是物理 |
| 拓扑 | 解析 `nvidia-smi topo -m` 的 NVLink/PIX/PXB/PHB/SYS 与 NUMA，喂给规划器；**在 `SYS` 拓扑上不给 ZeRO-3 背书** |

### 规划器会**拒绝**坏方案（相对 PyT 的差异点）

```python
[dist] 方案 dp / world=2
   rank0    8.74 GiB / 可用 29.79 GiB
   通信 6.79 GB ⇒ 约 140.9 ms（按实测 48.2 GB/s）
   吞吐比估算 0.99×（单卡 0.1372 s + 暴露 141 ms = 0.2781 s）
   ⚠️ 估算 < 1.00×：加卡是净亏
```

判据（数据并行每卡算自己的 batch，**步时不缩短**）：

```
吞吐比 = 2 × 单卡步时 ÷ (单卡步时 + 暴露的通信)
```

实测校准：5090 seq512 算 0.99× / 实测 0.87×；seq2048 算 1.26× / 实测 1.20×；
4090 seq512 算 0.68× / 实测 0.68×。

### 诚实边界（交付时不许越过）

* `strategy="tensor_parallel"` / `"pipeline_parallel"` **调用即拒绝** —— 引擎没实现，不假装支持
* 两个**活会话**共用一个 `job`+`dir`：已按会话号隔离（`ready.<token>` / 共享内存名 / barrier 名），
  但**未验证** —— 交付必须写"未验证"
* 无内核级内存隔离（只有 95% 显存预留 + OOM 关键字识别）
* `cudaIpcGetMemHandle` / SHM 直传梯度**未实现**（现在走 host 打包或 NCCL）
* **`TG_DP_OVERLAP` 默认关**（1.0.16 起）：重叠路径实测会静默不归约梯度
  （`comm=0.00ms` 却报 `pending_zeroed=True`），已有硬自检会当场 raise。详见
  `docs/OPEN_DP_GNORM_MISMATCH.md`
