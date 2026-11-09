# TGAI V2.7 全参训练：运行手册（V100-32GB）

> 状态：**技术验证全部完成，等待用户授权后再执行**。
> 本文是"点头即可照做"的操作清单，不含未验证内容。

## 0. 前置事实（均为实测）

| 项 | 数值 | 来源 |
| --- | --- | --- |
| 模型 | 3.033B / 983 张量（`.TG` 内 982，`lm_head` 与 embedding tied 不存） | 实测 |
| 权重显存（fp32） | 11.2~11.3 GB | `empty_gpu` + 逐层斜率双证 |
| fp16 动量 | 10.9 GB | 实测 |
| 逐层梯度 | ~0.1 GB（不累加） | 2 层实测，峰值 102 MiB |
| 训练峰值 | **24.8~25.9 GB / 32 GB** | 多次实测（T=256/512、300 步） |
| 速度 | **1.49~1.87 s/步**（T=256~512） | 实测 |
| 装载耗时 | 87~95 s（`.tg` 直读 982 张量） | 实测 |
| 训练有效性 | loss 29.07→0.88（同批 300 步）；真实语料 4.16→0.80 | 实测 |
| 质量基线 | 留出 CE 4.08；teacher forcing top-1 24~34%、top-5 46~57% | 实测 |

## 1. 训练命令（从产线 v4 直读）

```bat
python D:\tgai_orv\v27_train_run.py --model D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg ^
    --steps 200 --seq 512 --lr 1e-6
```

* 脚本自带**隔离**（停 8 个计划任务 + 杀 `ka_loop` + 改名 launcher）与**恢复**；
* `finally` 里先释放显存（轮询到 <2 GB）再恢复产线；
* 每 10 步 `empty_cache()`（实测把长跑显存稳在 23~25 GB）。

## 2. 必须先确认的三条纪律（都来自实测教训）

1. **11450 必须比 11451 先启动**（同时起会抢显存，11450 OOM）；
2. **lr ≤ 1e-5**：1e-4 实测发散（29→199）；
3. **训练前确认 GPU 空闲**：`nvidia-smi` < 100 MiB 且 `procs=0`，否则不要启动。

## 3. 检查点

* 已实现：`TGState`（读 `.tg`）+ `TGWriter.write_stream`（写 v4，**与产线逐位一致**，除 version 字段）
* 导出命令示例（训练后）：
  ```python
  w = TGWriter(...); w.write_stream(OUT, items(), version=4, tok_blob=<产线TOK>)
  ```
  关键规则：**rope 表与所有 norm 用 fp32，其余 bf16**；`tok_blob` 直接复用产线分词段。
* **已实现（2026-11-09 实测往返通过）**：优化器状态存盘/续训
  Fp16MomAdamW.state_dict() / load_state_dict() +
  save_opt_states(opts, dir) / load_opt_states(opts, dir)（**每层一个 layer_XXX.npz**，逐层落盘）
  实测：step → 存 → 清零 m/v 与步数 → 载回 ⇒ 	/m/v **全部一致**。
  ⇒ 长跑中断可**精确续训**（权重用 write_stream 存 v4，动量用 npz）。

### 3.1 磁盘规划（实测，2026-11-09）

`
D: 可用 61.6 GB
单个完整检查点 = 权重 6.06 GB + fp16 动量 10.90 GB = 16.96 GB
⇒ 最多存 2 个（还要留 20 GB 余量）
`

**开训前必须先清理**（待授权）：D:	gai_orv 下约 23 GB 测试产物
（export_v4_full.tg / export_v4_full2.tg / 	rained_v4.tg / export_test.tg 各约 5.6 GB，llvm.tar.xz 0.9 GB —— 均可重新生成）。

**建议策略**：轮转保存（权重 + 动量各留最新 2 份），并在每次存盘后检查剩余空间，低于 15 GB 就删最旧一份。

## 4. 产线切换（可选，独立于训练）

```bat
python D:\tgai_orv\switch_v4.py --dry-run   :: 先看计划（默认）
python D:\tgai_orv\switch_v4.py --apply     :: 执行（停 2~4 分钟）
python D:\tgai_orv\switch_v4.py --rollback  :: 回滚到 v3
```

## 5. 未完成 / 需授权

| 项 | 阻塞原因 |
| --- | --- |
| 正式训练启动 | 需用户确认（会占满整卡数小时） |
| 产线切 v4 | 需用户确认（服务短暂中断） |
| 优化器状态存盘 | 未实现（长跑必须） |
| DPO / GRPO | 用户明确要求"回来再讨论" |