# -*- coding: utf-8 -*-
"""补记真机 P100 的复测结论（必须纠正我们自己之前的暗示）。"""
import io

P = "docs/EXTERNAL_P100_2026-11-15.md"

ADD = '''## 22. 真机 P100 反馈：44 KB 那条**不成立**（2026-11-15 晚补记）

外部同学在**真 sm_60** 上复测，结论与我们的 V100 模拟相反：

| 项 | 结果 |
| --- | --- |
| 融合反向小形状（(2,64,64)、(28–48, 2048, 64)，含非连续输入、policy standard/speed）| **全部通过**，dq 1.57 / 2.6–4.0 |
| 全模型 v3exp2（330M, B=4, S=2048）| 失败：`cuFuncSetAttribute(attn_bwd_flash_w maxdyn) rc=1 invalid argument` |
| 关掉 lowmem（回退物化注意力）| OOM：B=4 活张量 11.86 GB + 申请 0.75 GB；B=8 13.15 GB |
| 同形状 torch | 4.52 s / 1,811 tok/s / **6.65 GB**（不 OOM）|

**必须承认的两点**

1. `attn_bwd_flash_w` 在 P100 上连属性都设不上（rc=1），所以「44 KB ≤ 48 KB 所以能跑」这个推断
   **是错的**。我们只在 V100 上**模拟**过 48 KB 上限（LIMIT_KB 48 触发切换），那验证的是
   **切换逻辑**，不是 **P100 的驱动行为**。CHANGELOG 里「真机 sm_60 未实测」这句当时写对了，
   但这里的表格给了「≤48 KB 即可」的暗示 —— 以本节为准。

2. v3exp2 的 13.00 GB / 2.72 s / 3,013 tok/s 是 **V100** 的数字，不是 P100。
   对方在 P100 上复现不出来是**预期之内**。对外表述必须带硬件前缀。

**新暴露的真问题（比 smem 更值得查）**：同一形状 torch 6.65 GB、我们 13.86 GB，约 **2 倍**。
不是 smem 的事，是我们同时持有的激活/中间量比 torch 多。

**下一步（按优先级）**

1. 把 DCHUNK 从 16 降到 8，动态 smem 从 45,056 B 降到约 22 KB，绕开 P100 的属性上限再测；
2. 向对方要两个数：设备属性 `cudaDevAttrMaxSharedMemoryPerBlockOptin`，以及 `TG_DEBUG=1`
   打出的实际申请量 —— 前者定上限，后者定我们算得对不对；
3. 查 2 倍显存差：统计一次前反向里同时存活的张量集合（`peak_live_bytes` 从不重置，
   只能一个变体一个进程测）。
'''


def main():
    s = io.open(P, encoding="utf-8").read()
    if "## 22. 真机 P100 反馈" in s:
        print("已存在")
        return 0
    io.open(P, "w", encoding="utf-8", newline="\n").write(s.rstrip() + "\n\n" + ADD)
    print("已补记第 22 节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
