# -*- coding: utf-8 -*-
"""阶段一发版：0.8.0。升版本（pyproject + __init__ 两处 fallback）→ 写 CHANGELOG → 构建 → 校验 wheel。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.8.0"

ENTRY = """## [0.8.0] - 2026-11-09

阶段一：训练基础设施对齐 PyTorch（每项都有与 torch 的数值对照测试）。

### 新增

- **学习率调度器** `T.optim.lr_scheduler`：`ConstantLR` / `LinearLR` / `CosineLR`（含 warmup）/
  `StepLR` / `LambdaLR` / `make_scheduler`。lr 序列与 `torch.optim.lr_scheduler` 逐 step 一致
  （float32 存储精度内，判据 1e-6）。
- **优化器** `T.optim.SGD`（momentum / dampening / nesterov / weight_decay）、`T.optim.Lion`、
  `T.optim.Adafactor`。
- **梯度裁剪**导出到 `T.nn.utils.clip_grad_norm_`（torch 同路径）与 `T.clip_grad_norm_`。
- **梯度累加** `T.optim.Accumulator`（tick/finalize/reset）与 `T.optim.scale_grads`。

### 与 torch 的实测对照

| 项 | 判据 | 实测 |
| --- | --- | --- |
| 调度器 lr 序列 | 1e-6 | 逐 step 一致（float32 存储精度）|
| SGD（5 组配置，4 步）| 1e-6 | ≤1e-6 |
| Lion（含 wd）| 1e-6 | ≤1e-6（对独立参考实现）|
| Adafactor（2D/1D/wd/scale_parameter/clip）| 1e-6 | ≤1e-6（对 `transformers.optimization.Adafactor`）|
| 梯度裁剪（3 组 max_norm）| 1e-6 | 范数与梯度均 ≤1e-6 |
| 梯度累加 4×(3 行) vs 1×12 行 | 1e-6 | **1.30e-06**（fp32 求和顺序，见下）|
| 激活重算 开/关 梯度 | 1e-5 | ≤1e-5 |

梯度累加未达 1e-6（实测 1.30e-06）：数学等价，差异来自 fp32 下 4 次小 GEMM 与 1 次大 GEMM 的
求和顺序。测试判据写明 1e-5 并记录实测值，未把"接近"当作达标。

### 修正的 torch 语义（对照测试抓出来的）

- **SGD 首步不做 dampening**：torch 用原始梯度初始化 momentum buffer，之后才乘 `(1-dampening)`。
- **`CosineAnnealingLR` 在 T_max 之后周期回升**（warm restart），不是单调降到 eta_min。
- **模块参数的 `p.data` 是 memoryview**：`p.data[...] = x` 会抛
  `memoryview: invalid slice key`，新优化器统一改用 `np.copyto(np.asarray(...))`。

### 显存（V2.7 3B，28 层，实测）

- AdamW（fp16 动量）：公式 **11.064 GB**，GPU 实测 **+11.072 GB**
- Adafactor（row/col）：公式 **0.021 GB**，GPU 增量小于 1 MiB（走了已预留的池）
- 约 **527×** 状态显存差距

"""


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def main():
    py = os.path.join(ROOT, "pyproject.toml")
    s = read(py)
    s = re.sub(r'(?m)^version\s*=\s*"[^"]+"', 'version = "%s"' % VER, s, count=1)
    write(py, s)

    ini = os.path.join(ROOT, "python", "tgai_go", "__init__.py")
    s = read(ini)
    s = re.sub(r'(?m)^(\s*__version__\s*=\s*")[^"]+(")', r"\g<1>%s\g<2>" % VER, s)
    write(ini, s)
    lits = re.findall(r'(?m)^\s*__version__\s*=\s*"([^"]+)"', read(ini))
    print("版本字面量:", lits)

    cl = os.path.join(ROOT, "CHANGELOG.md")
    txt = read(cl)
    if "[%s]" % VER not in txt:
        i = txt.find("## [")
        txt = (ENTRY + txt) if i < 0 else (txt[:i] + ENTRY + txt[i:])
        write(cl, txt)
        print("CHANGELOG 已插入", VER)

    for d in ("dist", "build"):
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            shutil.rmtree(p)
    for f in glob.glob(os.path.join(ROOT, "python", "*.egg-info")):
        shutil.rmtree(f, ignore_errors=True)

    r = subprocess.run([sys.executable, "-m", "build", "--outdir", "dist"], cwd=ROOT,
                       capture_output=True, text=True)
    print("build rc=%d" % r.returncode)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        return r.returncode

    wheels = glob.glob(os.path.join(ROOT, "dist", "*.whl"))
    print("产物:", [os.path.basename(x) for x in wheels])
    want = {"Adafactor": b"class Adafactor", "Accumulator": b"class Accumulator",
            "clip_grad_norm_ 导出": b"_clip_grad_norm_impl",
            "sched": b"class CosineLR", "checkpoint": b"def checkpoint("}
    ok = True
    for w in wheels:
        with zipfile.ZipFile(w) as z:
            names = z.namelist()
            meta = [n for n in names if n.endswith("METADATA")]
            txt = z.read(meta[0]).decode("utf-8", "replace") if meta else ""
            ver_ok = ("Version: %s" % VER) in txt
            print("wheel %s 版本正确=%s" % (os.path.basename(w), ver_ok))
            blob = b"".join(z.read(n) for n in names
                            if n.startswith("tgai_go/train/") and n.endswith(".py"))
            for k, needle in want.items():
                hit = needle in blob
                print("   %-18s 包含=%s" % (k, hit))
                ok = ok and hit
            ok = ok and ver_ok
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
