# -*- coding: utf-8 -*-
"""发布前的一次性整理：bump 版本号 + 在 CHANGELOG 顶部插入条目。

用 Python 而不是 PowerShell 写，是因为 CHANGELOG 是 UTF-8 中文，
PowerShell 的 Set-Content 在 GBK 控制台下会把内容写坏。
"""
import io
import re

NEW_VERSION = "0.9.64"

ENTRY = """## [0.9.64] - 2026-11-15

**三档性能目标全部达成**：真实训练步上，ORV 有一档同时赢下显存与速度。

### 改了什么

- **修掉 MoE 分派里最后一处主机往返**（`moe_topk_orv` 里的 `x[tok]`）。
  它的前向很便宜（0.074 ms），但反向是 `D2H + np.add.at + H2D`，每层一次 8 MB 往返。
  改用 `indexing.index_select`（反向走设备内核 `scatter_add_rows`）。
  整步 **0.40 s → 0.24 s**，检查点档 **0.76 s → 0.47 s**，显存不变。
- **`memory` 档的 `attn_tile` 从 128 改为 768**：实测显存 3,517 → 3,169 MiB、
  步时 2.35 → 0.81 s，**两轴同时更好**。
- **删掉策略里的死配置 `empty_cache_every`**：四组预设里都写过，但**没有任何代码读它**。
- 新增 `bench/`：逐模块微基准（CUDA event 计时 + 主机入队计时），一个模块一个文件，
  用来定位"是哪个模块慢、慢在 GPU 还是慢在主机"。

### 实测（V100 32GB，seq2048 / 8 层 / d512 / 8 头 / top_k=2 / 共享专家；PyT = 4,633 MiB / 0.507 s）

| 档位 | 显存 | 步时 | 相对 PyTorch |
| --- | --- | --- | --- |
| `speed` | 8,329 MiB | **0.24 s** | 速度快 2.1×，显存 1.8× |
| `extreme` | **3,692 MiB** | **0.47 s** | 显存 0.80×、速度 0.93× |
| `memory` | **3,169 MiB** | 0.81 s | 显存 0.68×、速度 1.6× |

### 数值与测试

- 分派对拍：前向 4.27e-07、输入梯度 2.86e-06
- 整模型 loss 轨迹与改前**逐位相同**
- 测试：18 failed / 760 passed / 26 skipped，与**改动前完全相同**
  （失败项全在 `clip_accum` / `layer_graph_budget` / `optimizer_parity` / `plugin_native`，
  不涉及 MoE 分派，属既有问题）

### 升级建议

不需要改代码，`T.set_policy(...)` 的档位名与参数名不变。
若你曾在自己的代码里读 `T.policy("empty_cache_every")`，请改为自己的变量 —— 这个键已删除。

"""


def main():
    # 1) 版本号（pyproject）
    p = "pyproject.toml"
    s = io.open(p, encoding="utf-8").read()
    s2 = re.sub(r'^version = "[0-9.]+"', 'version = "%s"' % NEW_VERSION, s,
                count=1, flags=re.M)
    if s2 == s:
        raise SystemExit("pyproject.toml 里没找到 version 行，未做修改")
    io.open(p, "w", encoding="utf-8", newline="\n").write(s2)
    print("版本 -> %s" % NEW_VERSION)

    # 2) 回退版本号（源码里没装包时的字面量）
    #    ★ 这一步以前漏过：CI 的 tests/test_version.py 直接抓到了
    #      "pyproject 是 0.9.64，回退值是 0.9.63"。别再靠人记。
    q = "python/tgai_go/__init__.py"
    t = io.open(q, encoding="utf-8").read()
    t2, n = re.subn(r'__version__ = "[0-9.]+"',
                    '__version__ = "%s"' % NEW_VERSION, t)
    if n == 0:
        raise SystemExit("没在 %s 里找到 __version__ 字面量" % q)
    io.open(q, "w", encoding="utf-8", newline="\n").write(t2)
    print("回退版本号改动 %d 处 -> %s" % (n, NEW_VERSION))

    # 3) CHANGELOG 顶部插入
    c = io.open("CHANGELOG.md", encoding="utf-8").read()
    if "[%s]" % NEW_VERSION in c:
        print("CHANGELOG 已有该版本，跳过")
        return 0
    io.open("CHANGELOG.md", "w", encoding="utf-8", newline="\n").write(ENTRY + c)
    print("CHANGELOG 已插入 %s 条目" % NEW_VERSION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
