# -*- coding: utf-8 -*-
"""0.9.35 发版：Muon 在 3B 不可用 —— 定论 + 删除无效的批量实现。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.35"
ENTRY = """## [0.9.35] - 2026-11-11

**定论：Muon 在 3B 规模不可用，且这是算法固有限制。** 两种优化都实测无效。

### 实测

| 配置 | 3B 整步 |
| --- | --- |
| Muon，`ns_fast=False`（顺序 FMA，与 OpenBLAS 逐位一致） | 62.9 s |
| Muon，`ns_fast=True`（Newton-Schulz 走 cuBLAS） | **35.0 s**（1.8×） |
| Muon + 按形状分组批量 bmm（672 个同形张量） | **35.0 s（零收益）** |
| AdamW（fp16 动量 + 逐层即时更新） | **0.709 s** |

### 为什么批量也救不了（关键测量）

批量实现把 NS 的算子数从 `30×G` 降到 30（G=8 时约 5× 减少），**步时完全没变** ——
说明瓶颈不是算子数量，而是**总 FLOP**：

- 每个 `[1024,4096]` 矩阵正交化约 **96 GFLOP**（5 步 × 3 次矩阵乘）；
- 3B MoE 有 224 个这样的矩阵 ⇒ 约 **21.6 TFLOP/步**；
- V100 fp32 峰值 15.7 TFLOP/s ⇒ 理论下限 **1.4 秒**，实测 35 秒（带宽受限）。

**即使实现完美，Muon 在 3B 上也至少比 AdamW 慢一个数量级。** 这不是代码问题。

### 处理

- 批量实现（`batch_ns`）**已删除** —— 实测零收益，不留无效复杂度；
- `ns_fast` 保留（真实 1.8× 改进），文档写明它救不了 3B；
- README 的 Muon 段改成上述定论：适合 < 1B，3B 及以上用 AdamW。

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
    import zipfile
    ok = True
    for f in files:
        if f.endswith(".whl"):
            with zipfile.ZipFile(f) as z:
                meta = [n for n in z.namelist() if n.endswith("METADATA")][0]
                body = z.read(meta).decode("utf-8", "replace")
                ok = ok and (("Version: %s" % VER) in body)
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                checks = {
                    "batch_ns 已删除": "batch_ns" not in core,
                    "_step_batched 已删": "_step_batched" not in core,
                    "ns_fast 保留": "ns_fast" in core,
                    "README 有定论": "Muon 在 3B 规模不可用" in body,
                }
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
