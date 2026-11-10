# -*- coding: utf-8 -*-
"""0.9.34 发版：Muon 新增 ns_fast（Newton-Schulz 走 cuBLAS，62.9 → 35.0 s/步）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.34"
ENTRY = """## [0.9.34] - 2026-11-11

Muon 新增 `ns_fast` 开关，3B 上步时从 **62.9 s 降到 35.0 s（1.8×）**。
**但仍比 AdamW 慢 49 倍，未到可用范围** —— 如实记录，不做可用性宣称。

### 根因修正（推翻上一轮的猜测）

上一轮把 62.9 s/步归因为"逐张量发射开销"，并按"按形状分组 bmm"去规划。
本轮读代码发现更具体的原因：`_newton_schulz5_gpu` 默认用的是
**顺序 FMA 内核**（`sgemm_mode(..., 1)`），**故意不用 cuBLAS** ——
为了让 GPU 结果与 NumPy/OpenBLAS 的 float32 逐位一致。

于是加了 `ns_fast=True`：同样的 5 步迭代，矩阵乘改走 cuBLAS。实测：

| 配置 | 3B 整步 | 峰值显存 |
| --- | --- | --- |
| Muon，`ns_fast=False`（默认，逐位可复现） | 62.9 s | 18.93 GiB |
| **Muon，`ns_fast=True`（cuBLAS）** | **35.0 s** | 19.17 GiB |
| AdamW（fp16 动量） | **0.709 s** | 24.78 GiB |

### 为什么只快了 1.8 倍（说明不是 GEMM 慢）

3B 有 2550 个 2D 张量，每个要走 5 步 × 约 6 个算子 ≈ **7.6 万次算子调用**。
按 0.4 ms/次估算正好是 30 秒量级 —— 所以瓶颈是**算子数量**，不是单个 GEMM 的速度。
换 cuBLAS 只解决了其中"GEMM 本身慢"的一部分。

**要真正可用，需要按形状分组批量做**（672 个同形张量可以一次 bmm 算完），
把 7.6 万次调用降到万级以下。这是下一阶段的工作，本轮未做。

### 结论（写进 README 前先记在这里）

- `ns_fast` 是**真实的 1.8× 改进**，可以用；
- 但 **Muon 在 3B 上仍不可用**（35 s/步 vs AdamW 0.709 s）；
- 任何"Muon 已可用于 3B"的说法都是错的。

套件：**567 passed / 28 skipped**。
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
                i0 = core.find("class Muon")
                mu = core[i0:] if i0 >= 0 else ""
                checks = {"ns_fast 参数": "ns_fast=False" in mu,
                          "ns_fast 传递": "fast=self.ns_fast" in core,
                          "cuBLAS 分支": "gemm_nt(x, x)" in core}
                for kk, hit in checks.items():
                    print("   %-14s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
