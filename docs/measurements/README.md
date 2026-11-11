# 测量原始证据

> 这些是仓库根目录原先散落的测量/排查输出，已整理到这里。
> **为什么保留**：`docs/ROADMAP_3B_PERF.md`、`docs/MOE_MODERN_ARCH.md`、README
> 里的数字都源自这些文件，删了就无法复核。
> **说明只写能从文件内容直接确认的内容**；标「未核实」的是我没能确认的。

## 关键证据

| 文件 | 内容 |
| --- | --- |
| `oom.txt` | PyTorch 28 层 MoE 1.997B 在 31.40 GiB 已分配后 OOM —— README 核心卖点的原始记录 |

## 性能

| 文件 | 内容 |
| --- | --- |
| `d2h.txt` | 去 d2h 同步的实测：4.183 vs 4.328 ms/层（结论：收益在噪声内） |
| `g512.txt` | T=512 图反向整步：min 754 / mean 780 ms |
| `mab.txt` | MoE 单层 fp32 11.823 ms vs AMP 10.214 ms |
| `mlb.txt` | 单层 MoE 前向 6.42 ms、前向+反向 13.12 ms |
| `mpb.txt` | 单层 MoE 各原语耗时（gate/up/down 等） |
| `q.txt` | PyTorch 侧 seq=512 amp 108.1 ms / 峰值 10.65 GiB |
| `sp.txt` | 前向链 28 层 234.3 ms；MoE 反向 28 层 329.8 ms |
| `sp2.txt` | 同上复测：237.0 / 332.4 ms |
| `tc.txt` | tensor core 探针：fp32@PEDANTIC 0.410 ms（20.9 TFLOP/s） |

## 显存

| 文件 | 内容 |
| --- | --- |
| `ad16.txt` | AdamW fp16 动量逐层更新：空闲 20.47 GiB / 峰值 11.39 GiB，2.970B |
| `m16.txt` | fp16 配置下 2550 张量 / 2.970B 的显存占用 |
| `mem.txt` | 28 层权重 11.39 GiB；空闲 20.47 / 31.86 GiB |
| `mu16.txt` | Muon fp16 动量下的显存与参数统计 |

## 数值

| 文件 | 内容 |
| --- | --- |
| `status_out.txt` | 各算子 CUDA-vs-CPU 相对差（sqrt 5.92e-08 等） |
| `stk.txt` | 逐层梯度校验：前向 1.77e-07 / dL-dx 1.17e-07 / 权重梯度最大 1.46e-06 |

## AMP

| 文件 | 内容 |
| --- | --- |
| `bf.txt` | fp32 基线与 amp 状态的对照（bf16=True fp16=False） |

## 失败排查

| 文件 | 内容 |
| --- | --- |
| `cg.txt` | 容量因子 GPU 校验曾因 min 被遮蔽而抛错的 traceback |
| `eq.txt` | 分组未覆盖参数（embed/final_norm/lm_head）的告警 |
| `gex.txt` | 符号探测：cublasSgemmGroupedBatched 存在、Ex 变体缺失 |
| `gfx.txt` | 分组 GEMM 返回 rc=7（unsupported） |
| `gfx2.txt` | 分组 GEMM rc=7 后的回退路径 |
| `mg.txt` | 8 次独立 GEMM vs 分组 GEMM 不可用的对照 |
| `p8.txt` | 某次切换的 traceback |
| `r7.txt` | 分组 GEMM 按标量签名修正 argtypes 后仍 rc=7 的逐项探测 |
| `t3a.txt` | 分组未覆盖参数告警（版本 A） |
| `t3b.txt` | Muon 分组未覆盖告警（版本 B） |
| `tmb.txt` | Muon 分组未覆盖告警 |
| `tmf.txt` | Muon 分组未覆盖告警（另一版） |

## 测试

| 文件 | 内容 |
| --- | --- |
| `tgf.txt` | pytest 单条结果：PASS test_grouped_helpers_exist |

## 训练日志

| 文件 | 内容 |
| --- | --- |
| `_m60.log` | 60M 级 LLaMA 试训（配置行可见 D=768 L=8） |
| `_m60e.log` | 空文件 |
