# -*- coding: utf-8 -*-
"""0.9.52：CUDA 数组索引 + 本轮的规划/实测文档。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.52"
ENTRY = """## [0.9.52] - 2026-11-12

**CUDA 上支持 `x[idx]` 数组索引**，并把本轮的规划与实测落成文档。

### 修了一个写代码就会撞的坑

`Tensor.__getitem__` 的 **CPU 路径本来就支持数组索引**（用 `np.add.at` 处理重复下标），
但 **CUDA 路径直接抛** `NotImplementedError: 切片暂不支持索引类型 ndarray`。
任何人在 GPU 上写 embedding 查表或 gather-based CE，第一次就会撞上。
写 `tools/h2h_clean.py` 时我连撞三次。

现在 CUDA 支持**首轴一维整型索引**（`x[idx]` / `x[idx, :]`）：

| 环节 | 实现 |
| --- | --- |
| 前向 | 1 维源用 `gather_flat`；多维源用 `gather_rows`（都是已有融合核）|
| 反向 | **主机中转** `np.add.at`，正确处理重复下标 |
| 负数索引 | 按 numpy 语义折回 |
| 越界 | 在主机侧立刻 `IndexError`，不下发到设备（避免算成错误位置）|

**关于反向走主机中转，说清楚**：它比融合核慢。这是刻意的取舍 —— 先保证正确、可用，
再谈快。**embedding 场景请继续用 `nn.Embedding`**（它有专门的融合核）。
本版只支持一维整型索引作用在首轴；多维花式索引与布尔掩码仍未支持。

新增 `tests/test_getitem_array.py`（8 条），**真 V100 上 8 passed**：
前向对 numpy、负索引、列表索引、重复下标梯度累加（必须是 2 不是 1）、越界报错、1 维源、
以及 CUDA 与 CPU 的逐位一致。

### 本轮同时落地的文档（不改代码）

| 文档 | 内容 |
| --- | --- |
| `ROADMAP_CUDA_TO_VULKAN.md` | **当前主计划**：7 个工作包 + 判据 + 依赖 + 「明确不做」清单 |
| `HARDWARE_INVENTORY.md` | 两台机器的**实测**能力（笔记本只有 Vulkan/AMD 核显；远端只有 CUDA/V100 且处于 **TCC**，Vulkan 枚举不到设备）|
| `BENCHMARK_H2H_2026-11-12.md` | 自包含头对头：dense 快 2.58× 但**显存多 1.89×** |
| `INDEX.md` 九点六 | **把 TG Link 与后端混用拆开**（前者是跨设备 P2P，后者是进程内多后端），并记录 TCC 结论 |

### 一个必须记住的更正

`_FP16_MAX_K` 默认不限制，但注释记录长 K 会让参数梯度误差恶化到 2e-2。
实测（V100，dense/fp32）：`TG_FP16=1` 让 18.63 ms → 16.85 ms（1.11×），该形状下 loss 差 ~1e-07。
**默认值未改**，作为待拍板项留在 ROADMAP §11。

套件：**670 passed / 40 skipped**。
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
        print(r.stdout[-500:], r.stderr[-500:])
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
                    "数组索引入口": "_int_index_array(" in core,
                    "CUDA 实现": "_array_index_cuda(" in core,
                    "越界检查": "数组索引越界" in core,
                    "负数折回": "np.where(arr < 0, arr + n, arr)" in core,
                }
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
