# -*- coding: utf-8 -*-
"""把 svs-ops 的实测数字同步进各处文档（避免文档里留过时声明）。

数字来源（V100，B=2、1×96000、6 档多分辨率 STFT 损失、前向+反向、fp32）：

| 版本 | ORV | PyTorch | 比值 |
| --- | ---: | ---: | ---: |
| 组合式（恒等核分帧） | 1690.7 ms | 65.6 ms | 25.8× |
| hop 块分解（逐帧循环） | 218.8 ms | 5.2 ms | 42× |
| + 单次 matmul | 213.6 ms | 5.2 ms | 41× |
| + 基矩阵张量缓存 | **171.9 ms** | **5.2 ms** | **33×** |
| 对拍 | 51 项全过，相对差 ~2e-07 | | |

注意两次 PyTorch 基准不同（65.6 → 5.2 ms）：前者与慢路径同进程、GPU 被前序 8.5 s 负载
拖慢；后者是同进程等量预热后的干净数字。对外只用 5.2 ms 这一组，并注明条件。
"""
import io
import os

REPL = [
    # (文件, 旧串, 新串)
    ("docs/EXTERNAL_SVS_2026-11-15.md",
     "| ORV（组合式插件） | **1690.7 ms** | 2.326434 |\n| PyTorch（cuFFT） | **65.6 ms** | 2.326434 |\n| 比 | **25.8× 慢** | 相对差 1.0e-07 ✓ |",
     "| ORV（组合式插件，恒等核分帧）| **1690.7 ms** | 2.326434 |\n"
     "| PyTorch（cuFFT）| **65.6 ms** | 2.326434 |\n"
     "| 比 | **25.8× 慢** | 相对差 1.0e-07 ✓ |\n\n"
     "> 后续已优化到 **171.9 ms vs 5.2 ms（33×）**，见 §3.1。"),
]

EXTRA = """
### 3.1 提速过程（每一步都实测）

| 改动 | ORV | 比值 |
| --- | ---: | ---: |
| 恒等核 `conv1d` 分帧（原始）| 1690.7 ms | 25.8× |
| hop 块分解（4 次切片 + 矩阵乘）| 218.8 ms | 42× |
| 段内改成**单次 matmul**（原为逐帧循环）| 213.6 ms | 41× |
| DFT 基矩阵改成**缓存张量**（不再每次 H2D）| **171.9 ms** | **33×** |

PyTorch 同条件基准：**5.2 ms**（与 ORV 同进程、等量预热后测；早期 65.6 ms 那组是同进程
跑在 1690 ms 慢路径之后、被前序 8.5 s 负载拖慢的数字，不作为对外口径）。

**为什么还差 33×**：组合式的每一步都是一次算子调用 —— 一次 STFT 约 40 个算子、
一次损失 12 个 STFT，正反向合计近千次调用；而 cuFFT 是一个内核。剩下的差距是
**调度开销**，不是数学量。要真正追平必须把 STFT/iSTFT 做成**带反向的原生算子**
（cuFFT 可以直接用机器上的 `cufft64_11.dll`）。
"""


def main():
    for path, old, new in REPL:
        if not os.path.exists(path):
            print("跳过（不存在）", path)
            continue
        s = io.open(path, encoding="utf-8").read()
        if old in s:
            s = s.replace(old, new, 1)
            if "### 3.1 提速过程" not in s:
                s = s.replace("\n## 四、", EXTRA + "\n## 四、", 1)
            io.open(path, "w", encoding="utf-8", newline="\n").write(s)
            print("已更新", path)
        else:
            print("锚点未命中", path)

    # PLUGIN_SPEC 与 SKILL 里的数字
    p = "docs/PLUGIN_SPEC.md"
    if os.path.exists(p):
        s = io.open(p, encoding="utf-8").read()
        s = s.replace("实测（V100）：元信息通过 → 12 个算子安装 → `stft_mag` 与 `torch.stft` 形状一致、\n相对差 **1.99e-07**；`cumsum` 正确。",
                      "实测（V100）：元信息通过 → **18 个算子**安装 → `stft_mag` 与 `torch.stft` 形状一致、\n"
                      "相对差 **1.99e-07**；`cumsum` 正确。\n\n"
                      "**已知性能缺口**：组合式 `stft_mag` 比 cuFFT 慢 33×（171.9 ms vs 5.2 ms，\n"
                      "B=2、L=96000、6 档分辨率、前向+反向）。差距来自算子调度次数（一次 STFT≈40 次调用），\n"
                      "不是数学量；追平需要把 STFT 做成带反向的原生算子（cuFFT）。")
        io.open(p, "w", encoding="utf-8", newline="\n").write(s)
        print("已更新", p)

    # 包描述里的对拍项数
    import json
    m = "examples/bundles/svs-ops/manifest.json"
    d = json.load(io.open(m, encoding="utf-8"))
    d["description"] = ("为歌声合成项目（DDSP + 残差向量量化 + 流匹配 GAN）补齐 ORV 缺失的音频算子。"
                        "全部为组合式实现，梯度由 autograd 自动给出。每个算子都与 PyTorch 做过逐项数值对拍"
                        "（tools/verify_svs_ops.py，51 项全过，相对误差 ~2e-07）。"
                        "已知性能缺口：组合式 stft_mag 比 cuFFT 慢 33×（171.9 ms vs 5.2 ms），"
                        "差距来自算子调度次数；追平需做成带反向的原生算子。")
    io.open(m, "w", encoding="utf-8", newline="\n").write(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
    print("已更新", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
