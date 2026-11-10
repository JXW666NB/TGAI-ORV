# -*- coding: utf-8 -*-
"""0.9.31 发版：文档收尾（核心卖点从算术变成实测）。仅文档与工具变更。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.31"
ENTRY = """## [0.9.31] - 2026-11-10

文档收尾：把核心卖点从算术变成实测。

- **PyTorch + 真 AdamW 实测 OOM**：纯 PyT 进程、28 层 MoE（1.997B，比 V2.7 更小）、T=512、fp32、
  开 gradient checkpointing。仅 fwd+bwd 峰值 14.96 GiB；建 AdamW（m/v 需 14.88 GiB）后走一整步
  得到 `OutOfMemoryError`（31.40 GiB 已分配，剩 14 MiB）。而 ORV 在 2.97B 上跑完 **709 ms / 24.78 GiB**。
- **头对头四格补齐**（T=128/512 × fp32/AMP）：ORV 快 1.11~1.16×；但 8 层规模上**显存是 PyT 更省**
  （ORV 多 0.7~3.5 GiB，主要来自 AMP 的 fp16 权重影子缓存）。旧 README 的 1.41~1.74× 已按新口径下修。
- 修掉对拍脚本自身的两个 bug（两边算式不一致、批量替换误伤 PyT block）。

本版仅文档与工具变更，库代码与 0.9.30 相同。
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
    print("产物:", [os.path.basename(f) for f in files])
    return 0


if __name__ == "__main__":
    sys.exit(main())
