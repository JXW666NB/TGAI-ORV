# -*- coding: utf-8 -*-
"""0.9.16 发版：C 端算子下沉（abs/relu/clamp/div）+ 高层 DPO/GRPO 训练器 + minimum/to()。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.16"

ENTRY = """## [0.9.16] - 2026-11-10

把最后 4 个"CUDA 上只能走 CPU"的算子沉到 C 端，并补高层偏好训练器。

### C 端内核（新增 7 个）

- `ewise_abs` / `ewise_abs_bwd`、`ewise_relu` / `ewise_relu_bwd`、
  `ewise_clamp` / `ewise_clamp_bwd`、`ewise_div`。
- 结果：`_cpu_only` 调用点从 4 个降到 **0**，源码里不再有"CUDA 上偷偷回退 CPU"的算子。
- 实测（V100 vs torch）：abs / relu / clamp 的**前向与反向均 0.00e+00**；log 前向 7.50e-08、反向 0.00e+00。

### 新增 API

- `Tensor.to(device)` / `.cuda()` / `.cpu()`：叶子级设备搬运（用于常量、输入、参考 logp）。
- `Tensor.minimum(other)`：此前**完全没有**；CUDA 上用 `0.5*((x+y)−|x−y|)`。
- `Tensor.maximum(other)` 的 CUDA 分支同样改为组合实现（原来走 `.data`，CUDA 上直接崩）。
- `DPOTrainer` / `GRPOTrainer`：高层训练器，负责冻结参考、清零、反向、裁剪、步进、统计。
  用户只提供 `(ids, mask) -> logπ`。

### 修复

- `align.grpo_loss` 在 CUDA 上会算出 **nan**：常量（old/ref/mask）没有对齐设备，且统计用了
  `.data`（CUDA 无主机镜像）。现在常量 `.to(p.device)`、统计走 `to_numpy()`。
  实测修复后 GPU 上 GRPO 一步 loss 有限、梯度范数 0.367。

### 实测（V100 smoke）

| 项 | 结果 |
| --- | --- |
| DPO 20 步（GPU） | loss 0.688 → **0.213**，margin 0.0995 → **14.398** |
| GRPO 一步（GPU） | loss 有限、adv 正确、裁剪范数 0.367 |
| abs/relu/clamp 前反向 | **0.00e+00** |
| CPU 侧 pytest | 508 passed / 20 skipped |
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
                cuda = z.read("tgai_go/train/cuda.py").decode("utf-8", "replace")
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                rl = z.read("tgai_go/train/rl_trainer.py").decode("utf-8", "replace")
                al = z.read("tgai_go/train/align.py").decode("utf-8", "replace")
                checks = {
                    "ewise_abs": "ewise_abs" in cuda,
                    "ewise_relu": "ewise_relu" in cuda,
                    "ewise_clamp": "ewise_clamp" in cuda,
                    "def minimum": "def minimum(self, other)" in core,
                    "def to(device)": "def to(self, device)" in core,
                    "DPOTrainer": "class DPOTrainer" in rl,
                    "GRPOTrainer": "class GRPOTrainer" in rl,
                    "grpo 设备修复": ".to(dev)" in al,
                }
                for k, hit in checks.items():
                    print("   %-16s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
