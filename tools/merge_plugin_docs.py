# -*- coding: utf-8 -*-
"""把 PLUGINS.md 与 PLUGIN_CHARTER.md 合并成一套说法。

做法：PLUGINS.md 保留「入门」角色（怎么写第一个插件、contrib、打包），
把其中的**规范内容**（规则、信任模型）改成指向宪法，避免两处各写一版而漂移。
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "docs", "PLUGINS.md")
s = io.open(P, encoding="utf-8").read()
orig = s
n = 0

# ── 1. 顶部加定位说明 ──
HEAD_OLD = "# 插件式算子（一切皆插件）\n"
HEAD_NEW = """# 插件式算子（入门）

> **本文是「入门」**：怎么写第一个算子插件、contrib 流程、怎么打包。
> **规范条款在 [`PLUGIN_CHARTER.md`](PLUGIN_CHARTER.md)** —— 那里写的是不变量、承诺、
> 兼容规则与安全边界，是**必须遵守**的部分。本文与它冲突时以宪法为准。
>
> 本书一套文档的分工：
> `PLUGIN_CHARTER.md`（规范）→ 本文（算子插件入门）→
> `PLUGIN_DISPATCH.md`（能力与分派）→ `PLUGIN_NATIVE.md`（C 端四通道）→
> `PLUGIN_DSL.md`（声明式内核，为"生成代码到 C 端"铺路）。

"""
if HEAD_OLD in s and "本文是「入门」" not in s:
    s = s.replace(HEAD_OLD, HEAD_NEW, 1)
    n += 1

# ── 2. 「规则」节 → 指向宪法 ──
RULE_START = "## 规则\n"
RULE_END = "## 官方 contrib"
i, j = s.find(RULE_START), s.find(RULE_END)
if i > 0 and j > i:
    s = s[:i] + """## 规则（规范条款见宪法）

这里只列入门必知的四条，**完整不变量在 [`PLUGIN_CHARTER.md`](PLUGIN_CHARTER.md) §4**：

* **不许静默覆盖核心算子**：重名直接报错；要替换核心实现须显式 `replace=True`，
  并会记入 `list_ops()`。
* **插件是用户代码**，引擎不审计其数值正确性 —— 正确性由插件自己的测试负责。
  建议照 `tests/test_op_plugins.py` 用 PyTorch 做对照。
* **插件不改变后端契约**：组合式插件自动适配任何后端；自带反向的插件走 numpy
  （任何后端都能跑，但绕过融合核）。
* **卸载**：`unregister(name)`。覆盖过核心的插件无法热恢复，会明确报错让你重启进程。

**关于性能的硬要求**：能由已有算子拼出来的算子，**不应该**只做 Python 逐算子实现
—— 见宪法 §3「生成优先、Python 兜底」。

---

""" + s[j:]
    n += 1

# ── 3. 「信任模型」节 → 指向宪法 ──
TRUST_START = "## 7. 信任模型（先说清楚，别误会）"
k = s.find(TRUST_START)
if k > 0:
    s = s[:k] + """## 7. 信任模型（规范条款见宪法 §7）

一句话版，**完整条款在 [`PLUGIN_CHARTER.md`](PLUGIN_CHARTER.md) §7**：

* 插件是**用户代码**，加载时被**直接执行**；引擎**不做沙箱**。只加载你信任的插件。
* `sha256` 只用于**溯源与复现**，不是签名验证，不能证明「这份代码安全」。
* `TGAI_OPS_STRICT=1` 管的是**契约合规**，不是安全。
* 想让插件跑在 GPU 上：组合式自动跟随后端；`register_numpy` 走 numpy，
  对 CUDA 可用但**绕过融合核**，性能自行评估。
* 加载原生库（`plugin.load_lib` / `op_c` / `op_native`）**等于执行任意机器码**，
  风险远高于纯 Python 插件 —— 见 `PLUGIN_NATIVE.md`。
"""
    n += 1

if s != orig:
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("合并处数:", n)
sys.exit(0 if n == 3 else 1)
