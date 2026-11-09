# -*- coding: utf-8 -*-
"""0.9.9 发版：LayerNorm + 图捕获前置修复 + 探针工具。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.9"

ENTRY = """## [0.9.9] - 2026-11-09

补齐 LayerNorm，并为 CUDA Graph（阶段 A2）做前置修复与诊断。

### 新增

- `T.nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`：用现有可微算子**纯组合**
  实现（无新内核）；CPU 与 CUDA 均已实测。
- `T.from_torch` / `T.to_torch` 支持 `nn.LayerNorm`（weight/bias/eps/elementwise_affine）。

### 实测

| 路径 | 前向 | 反向 |
| --- | --- | --- |
| CPU（对 `torch.nn.LayerNorm`）| ≤1e-5 | 对输入/γ/β 均 ≤1e-5 |
| **CUDA（V100）** | **4.38e-08** | **9.29e-07** |

互操作往返（含 LayerNorm 的 Sequential）参数**逐位相同**。

### 修复与诊断（为 A2 铺路）

- `Tensor.backward()` 结尾的 `rt.sync()` 在**图捕获期间**跳过（捕获中同步非法，rc=900）。
- 新增 `tools/graph_probe.py`：捕获期间一旦发生 `sync`/`d2h` 就打印调用栈。
  用它定位到 A2 的真正阻断点：**MoE 的 `_combine_cuda` 把每专家偏移量 D2H 回主机**
  （`train/__init__.py:4059`）以驱动 Python 侧的按专家循环。
- 结论（如实）：**逐层图捕获当前不可行**，因为 MoE 的算子是数据相关的（每专家行数不同，
  启动形状要靠主机侧偏移量）。可行方向是把 MoE 改成容量因子（capacity factor）+ 掩码的定形计算，
  或只对稠密部分做图捕获。仓库里 `graph_train.GraphStep` 对**稠密模型**已有 1.71× 实测先例。

### 版本策略

1.0 之前一律 `0.9.N` 递增；`1.0.0` 留给"单卡可替代 PyT 训练自定义模型"的里程碑。
"""

DOCS = """
## 40. LayerNorm 与图捕获诊断（0.9.9，2026-11-09）

### 40.1 LayerNorm

`T.nn.LayerNorm` 用现有算子纯组合：`mu=mean(x,-1)` → 中心化 → `var` → `1/sqrt(var+eps)` → `*γ+β`。
实测：CPU 前向/梯度（对 x/γ/β）≤1e-5；**CUDA 前向 4.38e-08、反向 9.29e-07**。
互操作（含 LayerNorm 的 Sequential）往返参数逐位相同。

### 40.2 A2（CUDA Graph）诊断结论

- 驱动与运行时管线可用：`graph_available / graph_arm / graph_begin` 全为 True。
- `Tensor.backward()` 结尾的 sync 在捕获期间已跳过（必需）。
- **真正阻断点**：`MoE._combine_cuda` 把每专家偏移量 D2H 回主机（`train/__init__.py:4059`），
  为了用 Python 循环按专家分组。捕获期间任何 D2H/同步都非法。
- 结构性原因：MoE 每个专家的行数**依赖输入**，启动形状必须由主机侧偏移量决定 ⇒
  逐层捕获不可行，除非改成容量因子 + 掩码的定形计算（会改变显存/数值口径，需要单独评估）。
- 这条 D2H 同时是**每步每层一次主机往返**（28 层/步），本身就是性能项，值得单独优化。

### 40.3 可复现

```
python tools/graph_probe.py      # 捕获期间探针，打印 sync/d2h 调用栈
python tools/graph_layer_test.py # 逐层捕获尝试（当前会 abort）
python tools/layernorm_gpu_check.py
```
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
        write(cl, (ENTRY + txt) if i < 0 else (txt[:i] + ENTRY + txt[i:]))
        print("CHANGELOG OK")
    doc = os.path.join(ROOT, "docs", "TRAIN_V27_ORV.md")
    d = read(doc)
    if "## 40." not in d:
        write(doc, d + "\n" + DOCS)
        print("docs OK")
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
                blob = b"".join(z.read(n) for n in z.namelist()
                                if n.startswith("tgai_go/train/") and n.endswith(".py"))
                for k, needle in {"LayerNorm": b"class LayerNorm",
                                  "graph probe": b"_capturing"}.items():
                    hit = needle in blob
                    print("   %-12s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
