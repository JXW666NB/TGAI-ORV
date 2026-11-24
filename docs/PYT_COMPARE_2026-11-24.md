# ORV vs PyTorch 实测对照（2026-11-24，V100 32GB，1.0.27）

> 目标（用户口径）：**比 PyTorch 更快、更省显存、数学对齐、loss 下降曲线 1:1**。
> 本文只放**可复现的实测**，并明确写出还不能做到的地方。

环境：V100 32GB / CUDA 12.x / ORV 1.0.27（`patch_v8` overlay）/ PyTorch 2.6.0+cu124
参照实现：`D:\TGAI-V3-1.5B\03_src\model_v3.py`（真 V3）
数据：`D:\orv_bench\data2048_v96000`（8 分片，vocab 96000，seq 512）

---

## 一、速度与显存（同一配置、同一份数据、20 步）

配置：d=1024 / 12 层 / 16 头 / E=4 / d_ff=3328 / vocab=96000 / seq=512 / batch=1 /
top_k=2 / csa=4 / AdamW / lr=3e-4。

| 指标 | ORV 1.0.27 | PyTorch 2.6 | 比值 |
| --- | ---: | ---: | ---: |
| 稳态步时 | **0.2167 s** | 0.281 s | **ORV 快 1.30×** |
| 峰值显存（nvidia-smi）| **14,621 MiB** | 17,397 MiB | **ORV 省 2,776 MiB（−16%）** |
| GPU 利用率 | 21.6% | 27.6% | ORV 主机侧更省（更偏 host-bound） |
| 参数量 | 848,645,257 | 同 | — |

口径说明（对 ORV 不利的两点，仍然赢）：
* PyTorch 侧用 **fp16 AMP**（更省显存、更快），ORV 侧用 **fp32**（`--amp off`）；
  ORV 若开 `--amp gemm` 在 seq512 上是 **0.1638 s/步**（见 `HOSTOPT_ROUND2` §13）。
* ORV 的稳态步时取训练器自报 `step_s_steady`；PyTorch 取第 19 步墙钟（前 2 步含 warmup/CUDA graph 预热）。

复现：`lab/rent/_mem_compare.py`（远端 `D:\orv_bench\_mem_compare.py`）。

---

## 二、数学对齐（同一份初始化权重）

方法：PyTorch 侧 `torch.manual_seed(1234)` 建模型 → `faithful_parity.weights_from_torch_state_dict`
把**整份权重**灌进 ORV 参数结构 → 同一批 token 上比前向 logits。

| 规模 | logits max_abs | max_rel | 判定 |
| --- | ---: | ---: | --- |
| d=1024 / 4 层 / 16 头 / E=4 / d_ff=2816 / seq=512 | 1.09e-05 | **3.0e-06** | 通过（fp32 累积顺序差异量级）|

即：**同一个模型、同一份权重，两侧前向数值一致到 3e-6 相对误差**（|logits|max≈3.6）。

---

## 三、loss 下降曲线（同初始化 + 同数据 + 同步数）

这是"1:1"最关键的一项。方法（`lab/rent/_loss_curve_parity.py`）：
1. PyTorch 建模型并用固定 seed 初始化；
2. 权重整份灌进 ORV；
3. 两侧喂**同一批 token**（同一个 rng 序列），各自 `loss → backward → AdamW(lr=3e-4, wd=0.1, clip=1.0)`；
4. 逐 step 记录两侧 loss 并对比。

配置：d=1024 / **4 层** / 16 头 / E=4 / d_ff=2816 / vocab=96000 / seq=512 / batch=1。

| 步数 | 首步 loss（ORV / PyT）| 首步相对差 | 逐 step 绝对差（中位 / 最大）| 相对差（中位 / 最大）|
| ---: | --- | ---: | --- | --- |
| 3 | 11.692571 / 11.691761 | **6.9e-05** | 0.00136 / 0.00282 | 1.2e-04 / 2.4e-04 |
| 20 | 11.692571 / 11.691761 | 6.9e-05 | 0.00487 / 0.01057 | 4.2e-04 / 9.1e-04 |
| 50 | 11.692571 / 11.691761 | 6.9e-05 | 0.01421 / 0.06215 | 1.2e-03 / 5.3e-03 |

解读：**首步几乎逐位一致（7e-5）**；随后差异随步数缓慢累积（50 步到 1e-3 量级），
属于 fp32 归约顺序 + 梯度实现细节造成的混沌性漂移，而不是系统性偏差
（两侧 loss 同步下降，无单向偏离）。原始逐 step 数据：
`docs/measurements/curve_20.json`、`docs/measurements/curve_50.json`。

---

## 四、局限（不藏）

1. **曲线对拍只做到 4 层**：脚本把 torch 与 ORV 两套 1.07B 模型同时放进一张 32GB 卡会 OOM；
   4 层是 `docs/V3_FAITHFUL_PARITY.md` 已确认的安全对照规模。12 层的速度/显存对比是**分开进程**测的（第一节）。
2. **ORV 生产训练器与"真 V3"的架构差异**：`v3_pretrain` 的前向**没有实现 AttnRes**、
   **没有 MoE post-norm**、global 层默认不开 QK-Norm（CSA 层内部默认开）。
   所以本次对拍在 ORV 侧显式打开 `qk_norm`、并让 PyT 侧关掉 AttnRes/post-norm，
   使两侧是**同一个模型**；这一条已在脚本注释与本节写明，生产训练的默认配置不受影响。
3. `forward_loss`（训练用前向）与 `forward_logits`（对拍用前向）在同一权重下 loss 仍差
   ~8e-4（前者省略了 AttnRes 等参数的使用），这是上述差异的直接后果，不是数值误差。
4. 速度对比是 **host-bound 区间**（GPU 利用率 21–28%）；配置放大、GPU 打满后 ORV 的优势会缩小，
   这一条与 `docs/FUSION_FINAL_2026-11-23.md` 的结论一致。
5. PyTorch 侧未开 `torch.compile`；若开，其绝对速度会提升（但编译开销与动态 shape 另算）。

---

## 五、复现命令（V100）

```bat
REM 速度 + 峰值显存（同口径，20 步）
D:\orv_pip_0962\Scripts\python.exe D:\orv_bench\_mem_compare.py

REM loss 曲线（同初始化、同数据）
set TG_ORV_OVERLAY=D:\orv_bench\patch_v8
C:\Python311\python.exe D:\orv_bench\_loss_curve_parity.py --d 1024 --layers 4 ^
    --E 4 --n_q 16 --d_ff 2816 --vocab 96000 --seq 512 --steps 50 ^
    --json-out D:\orv_bench\curve_50.json
```
