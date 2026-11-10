# -*- coding: utf-8 -*-
"""0.9.18 发版：T=512 图反向打通（自动显存预算）+ rc=700 根因结论。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.18"

ENTRY = """## [0.9.18] - 2026-11-10

T=512 的逐层图反向**打通**（此前 `rc=700`），并给出根因与自动预算。

### 根因（实测定位）

同样的脚本、同样的 T=512，只是捕获张数不同：

| 捕获张数 | arena 合计 | 结果 |
| --- | --- | --- |
| 4 | 1.9 GiB | ✅ 通过 |
| 14 | 6.6 GiB | ✅ 通过，整步 **1187 ms** |
| 16（自动预算选的） | 7.5 GiB | ✅ 通过，整步 **1165 ms** |
| 28 | 13.1 GiB | ❌ 重放后读梯度 `cuMemcpyDtoHAsync rc=700` |

⇒ **不是 T=512 的图机制有问题，而是 arena 太多把显存挤爆**（随后前向链/MoE/梯度需要工作集）。
独立驱动 28 张图（不跑 MoE）时是**逐位一致 0.00e+00** 的，进一步佐证。

### 新增

- `LayerGraphBackward.max_affordable(n_layers)`：按当前空闲显存估算能安全捕获多少张；
  规则保守——`空闲 × 0.5 − 2 GiB` 作为可用额度。
- `LayerGraphBackward.capture(xs, max_graphs=None)`：`None` 时自动截断；
  新增 `count` 属性与 `has(i)`，调用方对没有图的层回退逐算子。
- `tests/test_layer_graph_budget.py`：守住"只捕获 N 张 / `has()` 边界 / 重放与逐算子一致"。

### 实测（V2.7，28 层，seq=512）

| 路径 | 整步 | loss |
| --- | --- | --- |
| 逐算子 | 2038.5 ms | 13.20335865 |
| 图反向（自动 16 张） | **1165 ms（1.75×）** | 13.20335865 |

### 代价/边界

- 图数量受显存限制，深模型长序列下**不会**全部层都吃到图加速（本例 16/28 层）。
- 每张图仍然固定占 `arena_bytes`（默认 256 MiB，T=512 建议 480 MiB）。
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
                lg = z.read("tgai_go/train/layer_graph.py").decode("utf-8", "replace")
                for k, needle in {"max_affordable": "def max_affordable",
                                  "自动截断": "max_graphs=None",
                                  "has": "def has(self, i)",
                                  "count": "def count(self)"}.items():
                    hit = needle in lg
                    print("   %-14s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
