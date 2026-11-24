# 外部基准（第三方提供，原样收录）

这里的脚本**不是本项目写的**，是为了复现与核对而原样收录的对照材料，方便以后追查。

| 文件 | 来源 | 用途 |
| --- | --- | --- |
| `bench_p100_full.py` | 群友在 Tesla P100 上的 torch vs ORV 对照（2026-11-15 提供）| 复现"稠密路径 1.34×"的差距 |
| `README_p100.md` | 同上，作者的方法学说明 | 解释为什么用"稠密等价"形状 |

## 为什么收录它

它测的是一件我们不常测的事：**ORV 的稠密 FFN 路径**。
我们自己的基准（`bench/b1`–`b3`、`tools/probe_ablate.py`）都以 MoE 为主，
而这里把 MoE 换成稠密 FFN、21 层、d=1152、18 头、S=2048、B=4，
正好覆盖了我们没测过的形状。

## 它的结论（已复核，见 `docs/EXTERNAL_P100_2026-11-15.md`）

- P100 无 Tensor Core，**torch fp16 与 fp32 完全等价**（步时与显存都一样）⇒ fp32 是公平基线
- 同精度下 ORV 在 B=4 慢 1.34×、显存高 1.54×，且 B=8 OOM
- 三档在 P100 上无区别：融合注意力要 68 KB 动态 smem > sm_60 的 48 KB

## 运行方式（与我们自己的基准一致的做法：先清卡）

```bat
taskkill /f /im python.exe
set HB=4
set FP32=1
set PYTHONPATH=D:\orv_src\python
C:\Python311\python.exe -u bench\external\bench_p100_full.py both
```

> 注意：本仓库自己的基准都在 `bench/` 顶层；`bench/external/` 只放第三方材料，
> 不要往这里加我们自己的代码。
