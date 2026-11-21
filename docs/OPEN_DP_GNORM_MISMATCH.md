# 未解决的正确性问题：数据并行下各 rank 的 gnorm 不一致

**状态**：**未修复**（2026-11-21 记录）。这是本轮唯一一条**已知的正确性问题**，
比任何吞吐数字都重要 —— 先写清楚，别让它丢在对话里。

## 现象（`zero2` 在 2×4090 上实测）

`848M / seq512 / int8gpu / 20 步`，同一份配置跑三档并行：

| 配置 | rank0 与 rank1 的 gnorm 在 20 步里不同的步数 |
| --- | --- |
| `--zero 0`（纯 DP，走 `dist.all_reduce_grads` 的 host 路径）| **9 / 20** |
| `--zero 1`（优化器状态分片，**复用同一条 DP all_reduce**）| 8 / 20 |
| `--zero 2`（自己的 reduce_scatter 主机路径）| **0 / 20** |

样例（step7）：`6.786449495799199`（rank0） vs `6.786491934594803`（rank1），相对差 **6.3e-6**。

## 为什么这算 bug

`python/tgai_go/train/v3_pretrain.py:1625` 的注释写得很明确：

> ★ 必须在裁剪**之前**归约：否则 gnorm 只是本 rank 的范数

也就是说归约发生在 `clip_grad_norm_` 之前，**归约之后两个 rank 的梯度缓冲应当逐位相同**
⇒ gnorm 必须逐位相同。现在不同，说明两个 rank 归约出来的梯度**不是同一个数**。

`--zero 2` 自己是逐位一致的（0/20），可见问题不在"分片数学"，而在
**`dist.all_reduce_grads` 的 host 路径**（`--zero 1` 复用它，所以一起出错）。

## 我怀疑的三条（**都还没验证**，按可能性排序）

1. **d2h 之前没有设备同步**。host 路径是「GPU 内打包 → 分块 d2h → 共享内存」。
   如果打包内核与 d2h 之间缺一次同步（或两者不在同一条流上），
   某个 rank 可能把**还没写完**的梯度发出去 ⇒
   rank0 看到 `g0 + g1(略旧)`、rank1 看到 `g1 + g0(略旧)` ⇒ 两边和不同。
   这与观测到的 **6e-6 量级**吻合（只差最后几个 bit）。
2. **本地梯度本身不对称**。MoE 反向用 `atomicAdd`，rank0 与 rank1 的**本地**梯度
   在最后几位就不同。这对"求和"不致命（`a+b` 可交换），但会放大任何同步问题的后果。
3. **分块边界 + 中途 barrier** 让某些块在某个 rank 上读到上一块的数据。
   （打包缓冲是复用的，`pack` 在 d2h 之后又被当作对端数据的落点。）

## 怎么验证（下次有 GPU 时，≤10 分钟）

1. 2-rank `--zero 0`、4 步、`TG_DP_TIMING=1`，把**两个 rank 的 gnorm 与 `pack`/`d2h` 各阶段耗时**一起打出来
2. 在 d2h 之前插一次 `rt.sync()`（环境变量开关），再跑同样 4 步：
   - **如果 gnorm 变逐位相同 ⇒ 就是第 1 条**（缺同步）
   - 如果仍不同 ⇒ 转为查第 3 条（分块/chunk 设 1 块试一次）
3. 对照组：`TG_DP_GRAD_DTYPE=bf16`（bf16grad 的路径）与 `--zero 2`（zero2 的路径）
   都不经过这段 host 打包，若它们逐位一致，就把范围锁死在 host 打包路径

## 影响面（诚实评估）

* 影响：**同一个训练里，两张卡看到的"梯度范数"不完全一样**。裁剪阈值因此有一点点差异
  （相对 6e-6），对收敛的实际影响极小，但**"各 rank 逐位一致"这条性质被破坏了**，
  而它是查其他 bug 的基准线（这一轮已经靠它抓到过两次真问题）。
* 不影响：单卡、`--zero 2`、NCCL 后端路径（待确认）。

## 复现命令（在 2×4090 上，`/root/bench` 已备数据）

```bash
CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=gn \
  TG_DP_TIMING=1 python -m tgai_go.train.v3_pretrain <通用参数> --zero 0 --steps 4 \
  > /tmp/r0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=gn \
  TG_DP_TIMING=1 python -m tgai_go.train.v3_pretrain <通用参数> --zero 0 --steps 4 \
  > /tmp/r1.log 2>&1 &
wait
grep -h gnorm /tmp/r0.log /tmp/r1.log
```
