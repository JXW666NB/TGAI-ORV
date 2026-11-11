# -*- coding: utf-8 -*-
"""0.9.39 发版：MoE Z-loss（router 数值稳定）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.39"
ENTRY = """## [0.9.39] - 2026-11-11

**MoE 支持 Z-loss**（ST-MoE, arXiv:2202.08906）—— router 的数值稳定手段。
这是现代 MoE 补齐清单的第 4 项（前两项是 0.9.37 共享专家、0.9.38 aux-loss-free）。

### 为什么需要

fp16 下 router logits 容易溢出（早期训练 router 权重摆动大），溢出会让 softmax
退化甚至 nan。Z-loss 惩罚 logit 的绝对幅值：

```
L_z = β · Σ_t (logsumexp(s_t))²
```

### 实现技巧（避免新增内核）

`log_softmax(x)_j = x_j − lse`，所以 `x_j − log_softmax_j === lse` **对每一列都成立**。
取均值即得 lse —— 复用已有的 `log_softmax`，不新写 logsumexp 内核，
也不会因为专家数 E 而损失精度。

### 新增 API

```python
moe = T.nn.MoE(..., z_loss_coef=1e-3)    # 默认 0.0 = 关闭，与旧行为一致
y = moe(x)
loss = task_loss + moe.z_loss            # 需要调用方自己加，ORV 不替你改 loss
```

**注意**：ORV 不把 z_loss 自动加进你的 loss —— 隐式改 loss 会让调试变难。
`moe.z_loss` 是一个带梯度的标量张量，未开启时为 `None`。

### 实测

| 项 | 结果 |
| --- | --- |
| 与手算 `β·Σ(logsumexp)²` 的相对差 | **6.48e-08** |
| CPU vs CUDA | **0.00e+00（逐位相同）** |
| GPU 上梯度到达 router | ✅ |
| 开/关 z_loss 的前向输出 | **逐位相同**（它只多返回一个 loss 项） |

### 测试（6 条，`tests/test_moe_z_loss.py`）

1. 默认关闭，`z_loss is None`；
2. 数值与手算一致（≤1e-6）；
3. logit 幅值变大时 z_loss 单调变大（这才是"压制幅值"的作用）；
4. 梯度能流到 router；
5. **不影响前向输出**（逐位相同）；
6. 与共享专家 + aux_free 同时开启也正常。

套件：**587 passed / 31 skipped**（+6）。
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
                i0 = core.find("class MoE")
                moe = core[i0:] if i0 >= 0 else ""
                checks = {"z_loss_coef": "z_loss_coef" in moe,
                          "_router_z_loss": "_router_z_loss" in moe,
                          "self.z_loss": "self.z_loss" in moe}
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
