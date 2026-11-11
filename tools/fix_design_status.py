# -*- coding: utf-8 -*-
"""修正 DESIGN.md 里两处与既有文档冲突的错误陈述（第二版，修掉嵌套引号）。"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "docs", "DESIGN.md")
s = io.open(P, encoding="utf-8").read()
orig = s
n = 0


def rep(old, new):
    global s, n
    if old in s:
        s = s.replace(old, new, 1)
        n += 1
        return True
    return False


rep("""- 多卡 / DDP / FSDP：【未验证】，且当前无机器。
- Vulkan 后端：【未实现】。
- `ParameterGroup` 在 CUDA 上：【未实现】（原因见 §5.4）。""",
    """- 多卡 / DDP / FSDP：代码与执行手册已存在（`docs/DP_2GPU_RUNBOOK.md`，
  判据「2 卡 vs 1 卡 ≥ 1.7×」），但本机无第二张卡，**从未跑过真实双卡** ⇒【未验证】。
- Vulkan 后端：**已有可运行原型**（`docs/VULKAN_BACKEND.md`，2026-11-03）：
  AMD Radeon 核显 Vulkan 1.3.260 上 compute 前反向均跑通，对 numpy **逐位精确 0.000e+00**；
  但**比 numpy 慢 20~150×**，根因是「同步提交 + 回读」。
  ⇒ 状态是【已实现但不可用】，不是「零代码」。要有收益必须先做**数据常驻显存 + 单命令缓冲批处理**。
- `ParameterGroup` 在 CUDA 上：【已实现】偏移视图（0.9.48，`CudaRuntime.sub_view`），
  真 V100 四项校验全 0.00e+00。""")

rep("| **C. Vulkan 后端** | 主战场，本目标只做架构准备 | 未开工 |",
    "| **C. Vulkan 后端** | 主战场。原型已跑通且数值逐位精确，但「逐算子同步 + 回读」的架构"
    "导致比 numpy 慢 20~150×，需重做为常驻显存 + 单命令缓冲 | 【已实现但不可用】 |")

rep("| 多卡零代码 | 未来要补大块 | 先留接口，见 §7.2 |",
    "| 多卡代码已有但**从未在双卡上验证** | 承诺不了加速比 | "
    "机器到手后按 `docs/DP_2GPU_RUNBOOK.md` 30 分钟内出结论 |")

rep("| Vulkan 零代码 | 主战场还没开始 | 见 §6.1 清单，按「先通后快」推进 |",
    "| Vulkan 原型**架构不可用**（同步提交 + 回读，慢 20~150×） | 主战场不能直接沿用 | "
    "重做为常驻显存 + 单命令缓冲，见 `docs/VULKAN_BACKEND.md` §4 |")

rep("**【未验证】当前一行多卡代码都没有。** 本机单卡，云上环境也不可控。",
    "**【未验证】多卡代码与执行手册已存在（`docs/DP_2GPU_RUNBOOK.md`），但从未在真实双卡上跑过。**"
    " 本机单卡、云上环境不可控 —— 所以「有没有代码」与「能不能用」是两件事，本文只承认后者未验证。")

rep("## 11. 相关文档",
    """## 10.2 一条流程教训（真实发生过）

本文第一版把「Vulkan 零代码」「多卡零代码」写进了事实清单，**是错的** ——
`docs/` 里早有 `VULKAN_BACKEND.md`、`DP_2GPU_RUNBOOK.md`、`BACKEND_PORTING.md`、`PLUGINS.md`。

**教训：写任何「设计基线」之前，先读 `docs/INDEX.md`。**
仓库已有 60+ 篇文档，不先读就会写出与既有结论冲突的「权威」，那比不写更坏。

## 11. 相关文档""")

if s != orig:
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("改正处数:", n)
sys.exit(0 if n >= 5 else 1)
