# -*- coding: utf-8 -*-
"""清理过期文档：加时效声明、归档，不删除历史测量记录。"""
import io
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
done = []


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


# ── 1. GAP_VS_PYTORCH.md：加时效横幅，并标出哪些结论已过期 ──
p = os.path.join(ROOT, "docs", "GAP_VS_PYTORCH.md")
s = read(p)
BANNER = """> ## ⚠️ 时效声明（2026-11-11 追加）
>
> **本文是 2026-11-02/03 的测量快照，部分结论已过期。**
> 最新的能力清单见 **[`CAPABILITY_AUDIT.md`](CAPABILITY_AUDIT.md)**（脚本生成，可复核）。
>
> 已确认**过期**的结论（原文说"没有"，但仓库里已有）：
>
> | 原文结论 | 现状 |
> | --- | --- |
> | "无 `state_dict` / `nn.Module` 互转" | 已有 `python/tgai_go/train/interop.py` |
> | "DPO/GRPO 明确抛错" | 已有 `ppo.py` / `align.py` / `rl_trainer.py` |
> | "多卡未验证" | 代码已存在（`dist.py` / `dist_p2p.py` / `pipeline.py`）+ `DP_2GPU_RUNBOOK.md`，仍**未在真双卡验证** |
> | "缺 Vulkan 后端" | C 引擎侧已有原型（`docs/VULKAN_BACKEND.md`），Python 侧有 `ops_vk*.py` |
>
> **仍然有效**的部分：§1–§5 的同形状硬指标、§8 起的显存/速度实测、以及 §11–§22 的
> 排查叙事（含多次自我更正）—— 这些是本文最有价值的内容，保留原样。
"""
if "时效声明（2026-11-11 追加）" not in s:
    i = s.find("\n", s.find("# "))
    s = s[:i + 1] + "\n" + BANNER + s[i + 1:]
    write(p, s)
    done.append("GAP_VS_PYTORCH.md 加时效声明")

# ── 2. 归档旧 README ──
src = os.path.join(ROOT, "README_OLD_0.2.md")
dst_dir = os.path.join(ROOT, "docs", "archive")
if os.path.isfile(src):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.move(src, os.path.join(dst_dir, "README_OLD_0.2.md"))
    done.append("README_OLD_0.2.md → docs/archive/")

# ── 3. 给 STATUS*.md 与 ROADMAP_3B_PERF 加指向 ──
for name, note in (
    ("STATUS.md", "本文为快照；最新能力清单见 `CAPABILITY_AUDIT.md`。"),
    ("STATUS_2026-11-09.md", "本文为 2026-11-09 快照；最新能力清单见 `CAPABILITY_AUDIT.md`。"),
):
    q = os.path.join(ROOT, "docs", name)
    if not os.path.isfile(q):
        continue
    t = read(q)
    if "CAPABILITY_AUDIT" in t:
        continue
    j = t.find("\n", t.find("# "))
    t = t[:j + 1] + "\n> ⚠️ %s\n" % note + t[j + 1:]
    write(q, t)
    done.append("%s 加指向" % name)

print("完成 %d 项：" % len(done))
for d in done:
    print("  -", d)
sys.exit(0 if done else 1)
