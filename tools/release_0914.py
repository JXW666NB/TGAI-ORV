# -*- coding: utf-8 -*-
"""0.9.14 发版：Dropout / 池化 / BatchNorm + Module.train()/eval() + 审查文档。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.14"

ENTRY = """## [0.9.14] - 2026-11-10

补齐自定义模型常用层：Dropout、MaxPool/AvgPool（1d/2d）、BatchNorm（1d/2d），以及
`Module.train()/eval()`。另附 `docs/REVIEW_2026-11-10_vs_PyTorch.md` 全面审查。

### 新增

- `nn.Dropout(p)`：inverted dropout，训练按 `1/(1-p)` 缩放；CUDA 走 `h2d + _from_gpu` 落同设备。
- `nn.MaxPool1d/2d`、`nn.AvgPool1d/2d`：用 `gather_flat`（前反向已在阶段 B 与 torch 逐位一致）实现，
  **CPU 与 CUDA 同一路径**；`AvgPool` 等价 `count_include_pad=True`（torch 默认）。
- `nn.BatchNorm1d/2d`：两遍法（均值→中心化后方差），只用已有 `mean/mul/add`，CPU/CUDA 通用；
  训练更新 running 统计，`eval()` 用 running 统计。
- `Module.train(mode=True)` / `Module.eval()`：递归到子模块，`self.training` 默认 True。

### 实测（V100，与 torch 对照）

| 项 | 相对差 |
| --- | --- |
| MaxPool2d 前向（CUDA vs CPU / vs torch）与反向 | **0.00e+00** |
| AvgPool2d 前向 / 反向 | 4.76e-08 / 9.69e-08 |
| BatchNorm2d 前向（CUDA vs CPU）与 running 统计 | **0.00e+00** |
| Dropout（CUDA） | 训练均值 0.9990、零比例 0.301、eval 恒等 ✓ |
| CPU 侧 13 条 pytest（含 BN 反向、1d 变体、坏参数拒绝） | 全绿 |

### 已知边界（不隐瞒）

- `AvgPool` 只支持 `count_include_pad=True`；`MaxPool` 在 CUDA 上要一次 `to_numpy()` 取 argmax
  （**主机往返**），并列最大值取第一个。
- `BatchNorm` 只支持 `track_running_stats=True`；`affine=False` 可用。
- CUDA 池化索引用 Python 循环按形状构造，大形状有主机开销；不是融合实现。
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
    write(ini, re.sub(r'(?m)^(\s*__version__\s*=\s*")[^"]+(")', r"\g<1>%s\g<2>" % VER, read(ini)))
    print("版本:", re.findall(r'(?m)^\s*__version__\s*=\s*"([^"]+)"', read(ini)))

    cl = os.path.join(ROOT, "CHANGELOG.md")
    txt = read(cl)
    if "[%s]" % VER not in txt:
        i = txt.find("## [")
        write(cl, (ENTRY + "\n" + txt) if i < 0 else (txt[:i] + ENTRY + "\n" + txt[i:]))
        print("CHANGELOG OK")

    for name in ("dist", "build"):
        p = os.path.join(ROOT, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
    for f in glob.glob(os.path.join(ROOT, "python", "*.egg-info")):
        shutil.rmtree(f, ignore_errors=True)
    r = subprocess.run([sys.executable, "-m", "build", "--outdir", "dist"], cwd=ROOT,
                       capture_output=True, text=True)
    print("build rc=%d" % r.returncode)
    if r.returncode:
        print(r.stdout[-600:], r.stderr[-600:])
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
                blob = z.read("tgai_go/train/layers_extra.py").decode("utf-8", "replace")
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                for k, (src, needle) in {
                        "Dropout": (blob, "class Dropout"),
                        "BatchNorm2d": (blob, "class BatchNorm2d"),
                        "MaxPool2d": (blob, "class MaxPool2d"),
                        "Module.train": (core, "def train(self, mode=True)"),
                        "nn 导出": (core, "BatchNorm2d")}.items():
                    hit = needle in src
                    print("   %-14s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
