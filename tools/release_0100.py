# -*- coding: utf-8 -*-
"""阶段三发版：0.10.0（自定义模型示例 + QUICKSTART）。升版本 → CHANGELOG → 构建 → 校验。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.10.0"

ENTRY = """## [0.10.0] - 2026-11-09

阶段三：证明"能用 ORV 训**用户自己的模型**"（不依赖 `.tg`，TGAI 只是样例）。

### 新增

- `examples/custom_model_quickstart.py`：纯 CPU、秒级可跑的自定义 Transformer 训练示例，
  演示 `AdamW + CosineLR(warmup) + Accumulator + clip_grad_norm_` 的组合用法。
- `docs/QUICKSTART_CUSTOM_MODEL.md`：安装、建模型、训练循环、一致性与能力边界。

### 实测（同模型、同初始化、同超参）

```
ORV vs PyTorch 各训练 8 步：逐 step loss 最差相对误差 = 1.44e-07   （判据 1e-4）
```

测试：`tests/test_custom_model_parity.py`。

### 能力边界（写进 QUICKSTART，未隐藏）

- 已支持：autograd、7 种优化器、5 种调度器、梯度累加/裁剪/激活重算、预训练/SFT/DPO/GRPO、
  单卡 32 GB 训 3B 全参 AdamW（torch 同卡 OOM）。
- 未支持：多卡；conv 与 gather 等通用算子；内置 DataLoader；与 `torch.nn.Module` 互操作。
"""


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def main():
    py = os.path.join(ROOT, "pyproject.toml")
    write(py, re.sub(r'(?m)^version\s*=\s*"[^"]+"', 'version = "%s"' % VER, read(py), count=1))

    ini = os.path.join(ROOT, "python", "tgai_go", "__init__.py")
    write(ini, re.sub(r'(?m)^(\s*__version__\s*=\s*")[^"]+(")', r"\g<1>%s\g<2>" % VER, read(ini)))
    print("版本字面量:", re.findall(r'(?m)^\s*__version__\s*=\s*"([^"]+)"', read(ini)))

    cl = os.path.join(ROOT, "CHANGELOG.md")
    txt = read(cl)
    if "[%s]" % VER not in txt:
        i = txt.find("## [")
        write(cl, (ENTRY + txt) if i < 0 else (txt[:i] + ENTRY + txt[i:]))
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
        print(r.stdout[-1200:], r.stderr[-1200:])
        return r.returncode

    wheels = glob.glob(os.path.join(ROOT, "dist", "*.whl"))
    print("产物:", [os.path.basename(x) for x in wheels])
    ok = True
    for w in wheels:
        with zipfile.ZipFile(w) as z:
            names = z.namelist()
            meta = [n for n in names if n.endswith("METADATA")]
            ver_ok = (("Version: %s" % VER) in z.read(meta[0]).decode("utf-8", "replace")) if meta else False
            print("wheel %s 版本正确=%s" % (os.path.basename(w), ver_ok))
            ok = ok and ver_ok
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
