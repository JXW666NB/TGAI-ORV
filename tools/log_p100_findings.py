# -*- coding: utf-8 -*-
"""把 P100 报告的发现记入 DESIGN 技术债表与 INDEX。"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
n = 0


def rep(path, old, new, count=1):
    global n
    p = os.path.join(ROOT, path)
    s = io.open(p, encoding="utf-8").read()
    if old in s:
        s = s.replace(old, new, count)
        io.open(p, "w", encoding="utf-8", newline="\n").write(s)
        n += 1
        return True
    return False


# ── 1. DESIGN 技术债：补两条来自 P100 实测的债 ──
rep("docs/DESIGN.md",
    "| `ParameterGroup` 不支持 CUDA |",
    """| **梯度检查点显存回收弱于 PyTorch**（P100 实测：400M/16GB 上 torch 能跑、ORV OOM） | 直接削弱"省显存"这个核心卖点；16GB 卡上用不了 | 对齐 torch 的检查点回收；先做逐层激活生命周期审计 |
| **NVRTC 依赖 DLL 定位不全**（`nvrtc-builtins64_*.dll` 找不到） | 首装即失败，用户必须自己改 PATH 才能起步 | 已修（0.9.50：`nvidia/*/lib`+`bin` 纳入搜索，并同时写进 PATH） |
| **fp16 只覆盖前向 GEMM**，反向仍 fp32 | 无 Tensor Core 的卡（P100/Pascal）上收益远小于预期；而 torch autocast 前反向都覆盖 | 要么给反向也做 fp16，要么在文档里明确"fp16 不等于快" |
| `Tensor` 缺 `cos` / `sin` | RoPE 类模型迁移需 numpy 预生成表 | 加两个一元算子 |
| `ParameterGroup` 不支持 CUDA |""")

# ── 2. INDEX：登记 P100 评估报告 ──
rep("docs/INDEX.md",
    "## 十、测量原始证据",
    """## 九点五、第三方硬件评估

| 文档 | 说明 |
|---|---|
| ★ `REVIEW_P100_2026-11-12.md` | **P100 16GB 实测报告（外部用户）**：首装 NVRTC DLL 失败、400M/16GB 上 OOM（PyT 能跑）、fp16 只覆盖前向、RoPE/SDPA/MoE 迁移缺口。含复现步骤 |

---

## 十、测量原始证据""")

print("改动处数:", n)
sys.exit(0 if n >= 2 else 1)
