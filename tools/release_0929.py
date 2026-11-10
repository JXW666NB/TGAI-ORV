# -*- coding: utf-8 -*-
"""0.9.29 发版：Muon 单测 + parameters() 去重兼容性修复 + AdamW 3B 实测。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.29"

ENTRY = """## [0.9.29] - 2026-11-10

补齐 Muon 单测，修一个**影响所有绑定权重模型**的兼容性 BUG，并实测 AdamW 路线在 3B 上的显存。

### 修复：`Module.parameters()` 没有去重（绑定词嵌入会被更新两遍）

`tie_embedding`（GPT 类模型的常见做法）让 `embed.weight` 与 `lm_head.weight` 指向
**同一个 `Parameter`**，它在 `named_parameters()` 里出现两次。原 `parameters()` 原样返回两份
⇒ 优化器**每步把这个权重更新两遍**，且不报错。实测与"按对象身份更新一次"的结果差 **7.2e-03**。

PyTorch 的 `parameters()` 会去掉重复参数，本次对齐该语义。这个 BUG 影响所有使用绑定权重的
模型（不只是 Muon —— AdamW / SGD 同样中招）。

### 新增：Muon 单元测试（9 条）

`tests/test_muon.py` 覆盖上一版只能靠 GPU 实跑验证的行为：

- 状态**按形状**分配（不是每参数三份），并复核 README 里 2.97B 的 11.1 / 5.5 GiB 推算；
- `fp16_mom` **只把 2D 那部分**减半；
- CPU 更新路径（2D 走 Newton-Schulz、1D 走 Adam 式）；
- `fp16` 与 `fp32` 动量的结果接近（相对差 < 5e-3）；
- `parameters()` 去重 + 预设确实共享同一 `Parameter`；
- 分组覆盖面强制校验、`auto_rest` 补齐、`step_layers` 与 `step` 的差异在 fp32 噪声带内。

### 修正：分层更新**不是逐位一致**

上一版的说法过于乐观。实测残差约 **2e-06**（参数更新量级 ~1e-2，相对 ~2e-4）：
分层更新改变了参数更新顺序，Newton-Schulz 迭代上会累积 fp32 噪声。
测试判据因此取 ≤1e-5，并在注释里写明"这是同一算法的舍入差，不是逐位一致"。

### 实测：两条优化器路线都能在 32GB 上跑完 3B 全参

V2.7（28 层 / 2.97B / MoE），V100 32GB，逐层即时更新：

| 优化器 | 优化器状态 | 峰值显存 | 剩余 | 结果 |
| --- | --- | --- | --- | --- |
| **Muon**（fp16 动量） | 5.5 GiB | **18.93 GiB** | 12.93 GiB | ✅ |
| **AdamW**（fp16 动量） | 11.1 GiB | **24.40 GiB** | 7.46 GiB | ✅ |

Muon 多出 5.5 GiB 余量（可用于更长序列或更大 batch）；AdamW 通用性更好，适合稠密模型。

### 新增 `Fp16MomAdamW.state_bytes()` / `bytes_per_param()`

与 `Muon` 同名 API，便于在训练前判断"这个模型装不装得下"，而不是等 OOM 再猜。

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
                body = z.read(meta).decode("utf-8", "replace")
                ok = ok and (("Version: %s" % VER) in body)
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                omv = z.read("tgai_go/train/opt_fp16mv.py").decode("utf-8", "replace")
                i0 = core.find("    def parameters(self):")
                seg = core[i0:i0 + 700] if i0 >= 0 else ""
                checks = {
                    "parameters 去重": "seen" in seg,
                    "Muon step_layers": "def step_layers" in core,
                    "Muon _step_cpu": "def _step_cpu" in core,
                    "AdamW state_bytes": "def state_bytes" in omv,
                }
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
