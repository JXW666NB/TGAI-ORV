# -*- coding: utf-8 -*-
"""0.9.17 发版：CUDA 组卷积（groups>1）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.17"

ENTRY = """## [0.9.17] - 2026-11-10

CUDA 上的**组卷积**（`groups>1`），conv 面最后一个缺口。

### 新增

- `conv2d` / `conv1d` / `Conv2d` / `Conv1d` 在 CUDA 上支持 `groups>1`：
  按组切片 → 复用已与 torch 对照到 2.39e-07 的单组路径 → 沿通道维拼接。
- `groups<=0`、或 `groups` 不整除通道数时明确报错并给修法。

### 实测（V100 vs `torch.nn.functional.conv2d`）

| 配置 | fwd | dx | dw | db |
| --- | --- | --- | --- | --- |
| groups=2, s=1, p=1 | 1.11e-07 | 7.81e-08 | 1.55e-07 | 6.64e-08 |
| groups=2, s=2, p=1 | **0.00e+00** | 6.35e-08 | 6.33e-08 | 1.01e-07 |
| groups=4, s=1, p=1 | 7.38e-08 | 5.59e-08 | 2.11e-07 | 1.02e-07 |
| groups=4, s=2, p=1 | 1.01e-07 | 4.21e-08 | 6.95e-08 | 2.39e-07 |

判据 1e-5，全部通过。测试：`tests/test_conv_cuda.py` 新增 groups=2/4 的契约用例
（无 CUDA 时跳过，CPU 侧套件保持全绿）。

### 代价（C3）

组卷积按组串行执行 g 次 gather+matmul，**不是**融合的分组内核；大 groups、大 batch 时明显慢于 cuDNN。
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
                conv = z.read("tgai_go/train/conv.py").decode("utf-8", "replace")
                for k, needle in {"单组函数": "def _conv2d_cuda_single",
                                  "分组调度": "def _conv2d_cuda(x, w, bias",
                                  "通道拼接": "def _cat_channels",
                                  "dim=1": "dim=1"}.items():
                    hit = needle in conv
                    print("   %-10s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
