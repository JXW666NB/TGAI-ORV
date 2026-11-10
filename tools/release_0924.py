# -*- coding: utf-8 -*-
"""0.9.24 发版：注意力堆叠实现（T=512 整步 754 ms，达成 0.911 s 目标）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.24"

ENTRY = """## [0.9.24] - 2026-11-10

**T=512 整步 754 ms —— 达成 0.911 s 目标**（此前 1058–1098 ms）。根因是分头小 kernel。

### 根因

V2.7 的 attention 是**逐头循环 16 次**：每层约 80+ 次小 kernel
（16×(Linear + Linear + Linear + qh@kh.T + softmax + pr@vh + Linear)），
每次都不足以填满 V100；同时让每张 CUDA Graph 的 arena 膨胀到 459 MiB，
只捕得下 22–24/28 层。

（先前怀疑是 `permute` 真拷贝导致，实测**不是** —— `fwd_attn` 里根本没有 permute。）

### 修法：堆叠实现

每层用 `cat` 把 16 个 `[DH,D]` 权重拼成 `[D,D]`（反向经 `cat` 自动流回原 `Parameter`，
**不改权重布局、不影响 .tg 加载、不需要重训**），然后：

```
3 次全宽投影 GEMM  →  reshape/permute 成 [NH,S,DH]  →  1 次批量 SDPA（16 头一起）  →  1 次输出 GEMM
```

`fwd_attn` 默认走堆叠；`attn_impl="loop"` 可切回逐头版（保留做对照与回退）。

### 实测（V2.7 28 层，T=512）

数值等价（对照逐头实现）：前向 ≤5.47e-07、`dL/dx` ≤5.75e-07、权重梯度 ≤1.46e-06。

| 段 | 逐头 | 堆叠 |
| --- | --- | --- |
| 前向链 | 411.6 ms | **234.3 ms** |
| MoE 反向 | 347.9 ms | 329.8 ms |
| 注意力反向 | 638.4 ms | **208.0 ms** |
| 词表头 | 14.2 ms | 10.7 ms |
| 合计（逐算子） | 1412 ms | **783 ms** |
| 合计（开图） | 1058–1098 ms | **754 ms** |

单层注意力前向（seq=256）：**9.8 ms → 2.7 ms（3.66×）**。

⇒ 端到端：2038.5 ms（起点）→ **754 ms（2.70×）**，且**低于 0.911 s 目标 17%**。
loss 13.20335388（与逐头版 13.20335865 相差 3.6e-07 相对，属 fp32 求和顺序差异，非逐位一致）。

### 顺带修掉一个严重 BUG

`nn.CausalSelfAttention` 的 head 切分**用错了**（`reshape(B,S,H,Dh)` 直接 ravel 成 `[B*H,S,Dh]`
会把 head 轴与 seq 轴混在一起），实测与逐头参照相差 **9.61e-01**、且不报错、无测试覆盖。
已修为 `permute(0,2,1,3)` 后再 reshape，并加 `tests/test_causal_self_attention.py`（6 条，含 B>1/H>1
与反向梯度）。
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
                ok = ok and (("Version: %s" % VER) in z.read(meta).decode("utf-8", "replace"))
                v27 = z.read("tgai_go/train/v27_train.py").decode("utf-8", "replace")
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                _a = core.find("class CausalSelfAttention")
                _b = core.find("class Sequential", _a)
                csa = core[_a:_b] if (_a >= 0 and _b > _a) else ""
                checks = {
                    "堆叠实现": "def fwd_attn_stacked" in v27,
                    "默认堆叠": 'attn_impl", "stacked"' in v27,
                    "逐头保留": "def fwd_attn_loop" in v27,
                    "CSA 已 permute": "permute(0, 2, 1, 3)" in csa,
                }
                for kk, hit in checks.items():
                    print("   %-14s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
