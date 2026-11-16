# ORV 在 V100 上预训练 TGAI V3 1.5B —— 实测记录（2026-11-16）

> 这是 **ORV 引擎的验收记录**：它能不能真正训练 V3，而不是只能跑一步冒烟。

## 一、1.5B 真配置 + 真语料（无中断 4 步）

配置：`d=1024 / 24 层 / 16 头 / E=4 / d_ff=3328 / vocab=96000 / seq=512 / csa=4 / top_k=2 + 1 共享专家`
→ **1,500,681,490 参数**（`1632` 个张量）

数据：`D:\TGAI-V3-1.5B\01_corpus\packed512\s512`（99 shards / 202,005,504 tokens / 词表 id 范围 [10,95997] ⊂ 96000）

| 项 | 实测 |
| --- | ---: |
| loss | **11.716139 → 11.389412** |
| s/step（稳态）| **0.990** |
| tok/s | **517** |
| 峰值（池 live）| **27.08 GiB** |
| nvidia-smi 峰值 | 30,577 MiB |

> ln(96000) = 11.472，起点 11.716 与之相符（随机初始化）；4 步内降 0.33 nats。

## 二、checkpoint 与续训（**逐位一致**）

```
[ckpt] D:\orv_runs\_r_a\step_000004 rotated_out=1
[resume] ...step=3 params=639 opt=639 data(epoch,rank,offset)=(0,0,3) loaded=36.70s
```

| | step 3 loss |
| --- | ---: |
| A：不中断 | 11.38941193 |
| B：从 `step_000003` 续训 | 11.38941193 |
| **\|d\|** | **0.000e+00** |

续训恢复的四样东西全部正确：模型参数、优化器状态、数据位置 `(epoch,rank,offset)`、LR 调度。

## 三、checkpoint 体积问题与修复（0.9.88）

一个 1.5B checkpoint = params 6 GB + Adam m 6 GB + v 6 GB = **18 GB**。
`--keep 2` 滚动第三次就写满 D 盘（剩余 46 GB），报
`OSError: 3407872 requested and 0 written` —— 表头像代码 bug，实为磁盘满。

**修复**：Optimizer 的 m/v 默认按 **fp16** 落盘 ⇒ checkpoint **12 GB**（-33%），
`load_optimizer` 兼容两种 dtype 并升回 fp32。需要逐位复现时 `half=False`。

## 四、与 PyTorch 的公平对比（同配置，3:1 CSA，seq=256）

| 实现 | s/step | tok/s | 峰值 |
| --- | ---: | ---: | ---: |
| PyTorch fp32 | 1.0866 | 235.6 | 22,303 MiB |
| PyTorch **autocast fp16** | 0.9427 | 271.6 | 21,973 MiB |
| **ORV fp32** | **0.55** | **~465** | **20,100 MiB** |

⇒ ORV 比 **PyTorch 开 autocast** 还快 **1.71×**（比 PyT fp32 快 1.98×），显存少约 1.9 GB。

## 五、fp16 现状与结论（重要）

- ORV 目前只有 **fp16 GEMM**（fp16 输入 + fp32 累加 + fp32 输出），等于 PyTorch autocast；
  **不是全 fp16 训练**（无 fp16 参数/激活/梯度、无损失缩放、无主权重）
- 实测在这套模型上 **fp16 GEMM 反而更慢**：1.13B/seq512 下 738 → 627 tok/s（+18% 时间），
  1.5B/seq512 直接 OOM（fp32 已 27.08 GiB，再加 3 GB 权重影子就爆）
- 原因：**本模型在 V100 上是内核派发瓶颈，不是 GEMM 瓶颈**（只跑到 fp32 峰值的 32%）
- ⇒ 全 fp16 的收益要等"派发开销"降下来之后才存在

## 六、局限（不藏）

1. 只跑了 **4 步**，不是正式预训练（正式训练需按天计）
2. 没做 1.5B 长跑的 loss 曲线（4 步不足以看趋势）
3. `--faithful-v3` 的六项在生产几何 20 层下：**相对误差 4.87e-06 通过，绝对 1.77e-05 超 1e-5 判据 1.8×**
4. 未测 `csa_compress ≠ 4`、`attn_res_block ≠ 4`、bf16（V100 不支持）
