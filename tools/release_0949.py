# -*- coding: utf-8 -*-
"""0.9.49：Muon 提速 6×（BUG #3）+ 插件文档合并成一套。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.49"
ENTRY = """## [0.9.49] - 2026-11-11

**Muon 3B 整步 35.0 s → 5.78 s（6.05×）**，并**更正一个我说错的结论**。

### 之前的错判

我曾写「Muon 是 FLOP 受限、已到物理下限、没法修」。**这个结论不成立。**
实测拆解（`tools/muon_profile.py`，真 V100）：

| 项 | 耗时 |
| --- | --- |
| 裸 GEMM `[1024,4096]×[4096,1024]` | 0.815 ms（10.54 TFLOP/s）|
| 裸 GEMM `[1024,1024]×[1024,1024]` | 0.233 ms（9.20 TFLOP/s）|
| **`read_scalar(sumsq_sequential)`** | **101.258 ms ← 真凶** |
| 一轮 NS（3 GEMM + elementwise） | 1.143 ms |
| 完整 5 步 NS（单矩阵） | **111.406 ms** |

⇒ **91% 的时间花在「每个矩阵一次单线程顺序归约」上**，不是算力。
纯 GEMM 每矩阵只要 9.3 ms。

### 修法：顺序归约 → 并行归约

`sumsq_sequential` 是单线程 fp32 顺序累加（当年为了与 OpenBLAS 逐位一致）。
`fast=True` 本来就已放弃逐位一致（cuBLAS 单次约 3e-8），所以改用已有的
两趟并行归约 `sumsq_parallel`（`sumsq_partial` + `sumsq_final`）。

| 项 | 修前 | 修后 |
| --- | --- | --- |
| `read_scalar(sumsq_*)` | 101.258 ms | **0.062 ms（快 1633×）** |
| 单矩阵 5 步 NS | 111.406 ms | **10.128 ms** |
| 224 矩阵外推 | 25.0 s | **2.3 s**（纯 GEMM 下限 2.1 s，即下限的 110%）|
| **真 3B 整步（实测）** | **35.0 s** | **5.78 s（6.05×）** |

### 顺带一个**反直觉但重要的发现**

并行版**比顺序版更准**：

| 实现 | 420 万项求平方和（期望 1677.0546）| 与 float64 参考 |
| --- | --- | --- |
| `sumsq_parallel`（分块） | **1677.0546** | **完全相同** |
| `sumsq_sequential`（单累加器） | 1670.0114 | 差 **4.2e-03** |

fp32 单累加器在大 n 上丢了三位有效数字；分块求和误差更小。
所以这次改动**同时更快、也更准**。

### 现状评价（不含糊）

5.78 s/步 **仍比 AdamW fp16 的 709 ms 慢 8×**，所以 **3B 上仍建议用 AdamW**。
但「完全不可用」的标签不再成立：从 50× 落后缩到 8×，且已贴到纯 GEMM 下限。
Muon 更适合 <1B。

### 测试（4 条，`tests/test_muon_ns_numerics.py`，CUDA 专用）

`parallel` 对 float64 参考 ≤1e-5；`sequential` 的漂移记录在 ≤1e-2 内；
`fast=True` vs numpy 参考 ≤1e-5；`fast` vs `slow` 两条路径 ≤1e-5；
CPU 上的 Muon 不受影响。真 V100 上 **4 passed**。

套件：**660 passed / 33 skipped**。

---

## 文档合并：插件系统只有一套说法

之前 `PLUGINS.md`（0.7.x 的算子插件入门）与新增的四篇 `PLUGIN_*` 部分重复，
容易产生两套说法。现已明确分工，并在 `PLUGINS.md` 顶部与相关章节加了指向：

| 文档 | 角色 |
| --- | --- |
| **`PLUGIN_CHARTER.md`** | **规范**（不变量 / 承诺 / 兼容 / 安全 / 评审），冲突时以此为准 |
| `PLUGINS.md` | 算子插件**入门**（怎么写第一个、contrib、打包） |
| `PLUGIN_DISPATCH.md` | 能力模型与分派（实测开销、bind 纪律） |
| `PLUGIN_NATIVE.md` | C 端四通道 |
| `PLUGIN_DSL.md` | 插件语言与各后端代码生成 |

### 宪法新增第 3 条：生成优先、Python 兜底

这是为**插件语言**铺路的核心条款：

| 档 | 形态 | 谁把数学变成机器码 |
| --- | --- | --- |
| **B1 声明式内核** | 用户写数学（逐元素/归约/广播）| **ORV 生成各后端源码并编译** |
| B2 后端源码 | 用户写 CUDA C / GLSL / 厂商语言 | 该后端工具链 |
| B3 预编译库 | 用户编译好的 `.so`/`.dll` | 用户自己的工具链 |
| P 组合式 | 用已有算子拼（Python）| 不需要 |

硬要求：能由 B1 表达的自定义算子**不得**只提供 P 档；缺生成器的后端必须让该插件
**表现为不存在**，不得静默降级；P 档兜底**必须可统计**，禁止静默兜底。
另有一条待实现清单（§9）明确记录 `plugin.dump_code` 与 `plugin.stats` 尚未实现。
"""


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def main():
    write(os.path.join(ROOT, "pyproject.toml"),
          re.sub(r'(?m)^version\s*=\s*"[^"]+"', 'version = "%s"' % VER,
                 read(os.path.join(ROOT, "pyproject.toml")), count=1))
    ini = os.path.join(ROOT, "python", "tgai_go", "__init__.py")
    write(ini, re.sub(r'(?m)^(\s*__version__\s*=\s*")[^"]+(")',
                      r"\g<1>%s\g<2>" % VER, read(ini)))
    print("版本:", re.findall(r'(?m)^\s*__version__\s*=\s*"([^"]+)"', read(ini)))

    cl = os.path.join(ROOT, "CHANGELOG.md")
    txt = read(cl)
    if "[%s]" % VER not in txt:
        i = txt.find("## [")
        write(cl, (ENTRY + "\n" + txt) if i < 0 else (txt[:i] + ENTRY + "\n" + txt[i:]))
        print("CHANGELOG OK")

    for name in ("dist", "build"):
        q = os.path.join(ROOT, name)
        if os.path.isdir(q):
            shutil.rmtree(q)
    for f in glob.glob(os.path.join(ROOT, "python", "*.egg-info")):
        shutil.rmtree(f, ignore_errors=True)
    r = subprocess.run([sys.executable, "-m", "build", "--outdir", "dist"], cwd=ROOT,
                       capture_output=True, text=True)
    print("build rc=%d" % r.returncode)
    if r.returncode:
        print(r.stdout[-400:], r.stderr[-400:])
        return r.returncode
    pr = os.path.join(ROOT, "publish_ready")
    if os.path.isdir(pr):
        shutil.rmtree(pr)
    os.makedirs(pr)
    files = glob.glob(os.path.join(ROOT, "dist", "*"))
    for f in files:
        shutil.copy2(f, pr)
    ok = True
    for f in files:
        if f.endswith(".whl"):
            with zipfile.ZipFile(f) as z:
                meta = [n for n in z.namelist() if n.endswith("METADATA")][0]
                ok = ok and (("Version: %s" % VER) in z.read(meta).decode("utf-8", "replace"))
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                checks = {
                    "并行 sumsq 用于 fast": "rt.sumsq_parallel(x) if fast" in core,
                    "顺序版仍保留": "rt.sumsq_sequential(x)" in core,
                }
                for kk, hit in checks.items():
                    print("   %-22s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
