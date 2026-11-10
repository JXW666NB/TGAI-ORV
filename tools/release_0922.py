# -*- coding: utf-8 -*-
"""0.9.22 发版：图捕获失败必须 abort（修复流卡死）+ T=512 图预算调到最优带。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.22"

ENTRY = """## [0.9.22] - 2026-11-10

修一个会让 GPU **后续操作全部失败**的 BUG，并把 T=512 的图预算调到实测最优带。

### 修复：捕获失败没有 abort，流会一直停在 capture 状态

`LayerGraphBackward.capture()` 在 `graph_begin()` 成功之后若中途抛异常（例如 arena 用尽），
没有调用 `graph_abort()`。结果是 CUDA 流永久停在"捕获中"，之后任何
`cuCtxSynchronize` / `empty_cache` 都返回
`rc=900 operation not permitted when stream is capturing` ——
**整台机器后续所有 CUDA 操作都失败**，而报错信息完全看不出根源。

实测复现：一次 arena 用尽之后，接着的每一次捕获尝试（哪怕参数完全正确）都失败在 rc=900。

现在用 `try/except` 包住，失败必 `graph_abort()`。修好后同样的探测脚本能连续报出每个配置的真实结果。

### T=512 图数量边界（实测，V2.7 28 层，每张 480 MiB）

| 捕获张数 | arena 合计 | 整步 |
| --- | --- | --- |
| 16 | 7.5 GiB | 1165 ms |
| 20 | 9.4 GiB | 1140 ms |
| **22-24** | 10.3-11.3 GiB | **1058-1098 ms** |
| 26 | 12.2 GiB | 1311 ms（分配器抖动，反而慢） |
| 28 | 13.1 GiB | 失败 `rc=700` |

自动预算改为 `空闲 x 0.65 - 2 GiB`，在 28 层/T=512 上落在 22-24 张这一档。
**T=512 整步：2038.5 ms -> 约 1058-1098 ms（1.86-1.93x）**，loss 与 eager 完全一致（13.20335865）。

### 未达成的目标（如实记录）

判据里的 **"T=512 整步 <= 0.911 s" 未达成**。整步分解显示注意力反向占 638 ms（45%）；
28 层全上图的估算是 862 ms（可达标），但每张图 arena 需 459 MiB，受压只能捕 22-24 层。

**进一步的方向已经定位**：CUDA 上 `permute/reshape` 是**真拷贝**而非视图，
使 arena 有约 3 倍冗余（理论上这一层只需 ~100-150 MiB，实测 459 MiB）。
视图化后 28 张约 4.4 GiB 可全部捕获，预计落到 862 ms 附近。
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
                lg = z.read("tgai_go/train/layer_graph.py").decode("utf-8", "replace")
                checks = {"graph_abort": "rt.graph_abort()" in lg,
                          "fraction 0.65": "fraction=0.65" in lg}
                for kk, hit in checks.items():
                    print("   %-14s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
