# -*- coding: utf-8 -*-
"""0.9.11 发版：多图 CUDA Graph 支持 + v27 前向拆分。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.11"

ENTRY = """## [0.9.11] - 2026-11-10

CUDA Graph 多图支持（逐层捕获/重放），以及 `V27Trainer` 前向拆分。

### 新增（图路径）

- `CudaRuntime.graph_count()` / `graph_launch_slot(i)` / `graph_release_all()`：可同时持有并重放**多张**图。
- 捕获期分配**只走 arena**、不再从池取（`_alloc`），`free()` 跳过 arena 指针（`_in_arena`）：
  否则图节点引用的缓冲会被后续捕获/复用/`empty_cache` 改写或释放（实测表现为第二张图结果不对、
  或 `cuMemCtxSynchronize rc=700`）。
- `reserve_capture_arena()` 支持**多块 arena 堆叠**（一图一块），`release_capture_arena()` 统一回收。
- `_graph_destroy()` 在有 slot 引用时不再释放 arena / exec。

### 新增（模型）

- `V27Trainer.fwd_attn(x, i, cos, sin)`：`ln1 + 16 头注意力 + 残差`，**不含 MoE**（无 D2H，可捕获）。
- `V27Trainer.moe_part(y, i)`：`ln2 + MoE + 残差`（保持逐算子）。
- `fwd()` 现在是 `moe_part(fwd_attn(x))`，算术不变。

### 实测（V2.7，seq=128，单卡 V100）

| 项目 | 数值 |
| --- | --- |
| 单层 attention 图重放 vs 逐算子 | 19.32 ms → **4.49 ms（4.3×）** |
| 28 层 attention 图重放 | 1016.3 ms → **140.5 ms（7.24×）**，梯度合计 **0.00e+00** |
| **真实 28 层反向阶段**（含每步梯度写回） | 952.6 ms → **159.8 ms（5.96×，省 793 ms）**，梯度 **0.00e+00** |
| 每图 arena 占用 | 约 116.5 MiB（256 MiB 预留足够） |

### 边界与未完成（如实）

- 图重放**逐位一致**的前提是：同一图、同一缓冲地址；上游梯度每步用 `cuMemcpyHtoD` 写入固定缓冲，
  已验证"换缓冲后结果随之改变、且与 eager 逐位相同"。
- **训练循环尚未接线**：`v27_train_run.py` 目前仍全逐算子。接线做法已写成
  `tools/graph_28_backward.py`（前向 eager、反向逐层重放），整步 loss 对照与 event 计时待做。
- 图路径**默认不启用**；不启用时行为与 0.9.10 完全一致。
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
                for k, needle in {"graph_launch_slot": b"def graph_launch_slot",
                                  "graph_release_all": b"def graph_release_all",
                                  "fwd_attn": b"def fwd_attn",
                                  "moe_part": b"def moe_part",
                                  "_in_arena": b"def _in_arena"}.items():
                    hit = needle in blob
                    print("   %-20s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
