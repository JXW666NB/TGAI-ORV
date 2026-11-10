# -*- coding: utf-8 -*-
"""0.9.13 发版：CUDA conv2d / conv1d（阶段 C2 收口）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.13"

ENTRY = """## [0.9.13] - 2026-11-10

CUDA 上的卷积（阶段 C2 收口）：**没有新写 CUDA 内核**，用 `gather_flat` + matmul 组合实现。

### 新增

- `conv2d` / `conv1d` / `Conv2d` / `Conv1d` 支持 **CUDA 张量**（`groups=1`）。
  `groups>1` 在 CUDA 上明确抛 `NotImplementedError` 并给修法，不静默算错。
- `_im2col_index_np`：主机侧构造 im2col 平坦索引（列序与 CPU 实现一致）。

### 实测（V100，与 `torch.nn.functional` 对照）

| 项目 | 最差相对差 |
| --- | --- |
| conv2d 前向 / dx / dw / db（6 组配置，含 padding、stride=2、dilation=2） | **2.39e-07** |
| conv1d 前向 / dx / dw / db（3 组配置，含 stride=2、dilation=2） | **1.26e-07** |
| groups>1 | 正确报错 |

判据 1e-5，全部通过。索引本身先用主机 `_im2col` 的输出作基准比对 6 组配置（0.00e+00）。

### 性能代价（C3，如实说明）

- 多一份补丁矩阵（`N*OH*OW*C*kh*kw`，与主机 im2col 同量级）
- `gather_flat` 是**逐元素** gather，不是融合卷积
- 索引在主机侧用 Python 循环构造，大形状有主机开销
- ⇒ **大 kernel / 大 batch 时明显慢于 cuDNN**。这是"可用且可验证"的实现，不是最优实现。

### 测试

- `tests/test_conv_cuda.py`：4 组形状/参数的前向与 dx/dw/db 契约 + groups 拒绝；
  无 CUDA 时自动 skip（CPU 侧套件保持 479 passed / 17 skipped）。
- `tools/conv_index_check.py`、`tools/conv_cuda_parity.py`、`tools/conv_cuda_extra.py` 为可复跑的对照脚本。
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
                blob = z.read("tgai_go/train/conv.py").decode("utf-8", "replace")
                for k, needle in {"_conv2d_cuda": "def _conv2d_cuda",
                                  "_im2col_index_np": "def _im2col_index_np",
                                  "cuda 分支": 'device", "cpu") == "cuda"'}.items():
                    hit = needle in blob
                    print("   %-18s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
