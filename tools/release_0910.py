# -*- coding: utf-8 -*-
"""0.9.10 发版：embedding 设备化 API + 诊断工具。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.10"

ENTRY = """## [0.9.10] - 2026-11-09

embedding 路径设备化（省 160 ms/步，loss 逐位一致），以及若干测量工具。

### 新增 API

- `V27Trainer.x0_from_ids(ids)`：用 `gather_rows` 在设备上构造 `x0`，不再把整张 `[T,D]` 传一遍主机。
- `V27Trainer.add_embed_grad_device(id_buf, g_buf)`：`emb_bwd` 后与 `head.weight.grad` 在设备上相加，
  不再把 `[V,D]`（V2.7 下 128 MB）D2H 相加再 H2D。
- `V27Trainer.forward_all_dev(x0_buf, cos, sin)`：从设备缓冲起步的逐层前向。

### 实测（同一 28 层 3B、seq=128、其余条件完全相同）

```
loss      numpy 路径 14.45316887   设备路径 14.45316887   相对差 0.00e+00
单步时间   numpy 1.158 s            设备 0.998 s           省 160 ms（13.8%）
```

省下的时间与校准对照自洽：旧路径有三次 ~128 MB pageable 传输（约 4 GB/s、每次约 40 ms）。

### 工具（诊断用，非库 API）

- `tools/kernel_count.py`：按内核名统计一层前向/反向的发射次数
  （实测 208 + 334，×28 = 15,176 次/步；elementwise 家族占前向 140/208、反向 261/334）。
- `tools/moe_d2h_cost.py`：MoE 主机往返成本（实测 0.062 ms/层、1.7 ms/步）。
- `tools/zeros_check.py`：校准 memset / h2d / 清零语义（用于判断计时是否可信）。
- `tools/full_step_embed.py`、`tools/embed_timing2.py`、`tools/graph_probe.py`、`tools/graph_layer_test.py`。

### 未完成（与 0.9.9 相同）

A2 图捕获（阻断点：MoE 的 D2H）、GPU conv、单步 ≤1.0 s（当前约 1.20 s）。
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
                for k, needle in {"x0_from_ids": b"def x0_from_ids",
                                  "add_embed_grad_device": b"def add_embed_grad_device",
                                  "forward_all_dev": b"def forward_all_dev"}.items():
                    hit = needle in blob
                    print("   %-22s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
