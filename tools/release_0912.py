# -*- coding: utf-8 -*-
"""0.9.12 发版：逐层图反向 API + 阶段 A 收口。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.12"

ENTRY = """## [0.9.12] - 2026-11-10

逐层 CUDA Graph 反向 API（实验性，默认关闭），阶段 A 收口。

### 新增

- `tgai_go.train.layer_graph.LayerGraphBackward`：按层捕获 attention 段（含重算前向）的反向图，
  训练时逐层重放。含 `capture(xs)` / `replay_layer(i, gw)` / `grads_numpy()` / `close()`。

### 实测（V2.7，seq=128，V100）

| 项目 | 数值 |
| --- | --- |
| 单层 attention | 19.32 ms → 4.49 ms（**4.3×**） |
| 28 层 attention | 1016.3 ms → 140.5 ms（**7.24×**），梯度合计 0.00e+00 |
| 真实 28 层反向阶段 | 952.6 ms → 159.8 ms（**5.96×**），梯度 0.00e+00 |
| 整步（同 harness） | 1843.2 ms → 721.3 ms（**2.56×**），loss/梯度 0.00e+00 |
| **生产形状整步** | **1151.2 ms → 664.4 ms（1.73×）**，**目标 ≤1.0 s 达成** |
| API 自测（2 层，参数梯度 + `dL/dx`） | **0.00e+00** |

### 必须知道的边界

- **优化器不在上面的生产形状计时内**：`Fp16MomAdamW` 的 fp16 动量占 11 GiB，与 28 张图的 arena
  同开会 OOM（`live 24.22 GiB` / 31.86 GiB）。优化器单独实测 **47 ms**，合计约 **711 ms**，仍在目标内。
- 图路径**默认不启用**；不启用时行为与 0.9.11 一致。
- `v27_train_run.py` **尚未接线**（`LayerGraphBackward` 已具备，接线是几行的事）。
- 逐位一致的前提：同一图 + 固定缓冲 + 每步 `cuMemcpyHtoD` 写入上游梯度；已实测"换缓冲→结果随之改变"。
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
                blob = b"".join(z.read(n) for n in z.namelist()
                                if n.startswith("tgai_go/train/") and n.endswith(".py"))
                for k, needle in {"LayerGraphBackward": b"class LayerGraphBackward",
                                  "replay_layer": b"def replay_layer",
                                  "graph_launch_slot": b"def graph_launch_slot"}.items():
                    hit = needle in blob
                    print("   %-20s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
