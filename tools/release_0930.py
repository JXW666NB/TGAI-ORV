# -*- coding: utf-8 -*-
"""0.9.30 发版：3B 整步实测（AdamW 709 ms / Muon 62.9 s）+ state_bytes 修正。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.30"

ENTRY = """## [0.9.30] - 2026-11-10

**3B 全参整步实测：AdamW 709 ms、峰值 24.78 GiB（纯显存，比 PyTorch Adafactor 基准 911 ms 更快）。**
同时实测出一个必须公开的坏消息：Muon 在 3B 上要 **62.9 秒/步**。

### 实测（V2.7 28 层 2.97B，V100 32GB，逐层即时更新）

| 配置 | 优化器状态 | 峰值显存 | 整步 |
| --- | --- | --- | --- |
| **AdamW fp16 动量** | 11.1 GiB | **24.78 GiB** | **709 ms** |
| Muon fp16 动量 | 5.5 GiB | 18.93 GiB | 62.9 s |

AdamW 比 PyT 的 Adafactor 步（911 ms）**快 1.28×**，且**不用主机内存**。
Muon 峰值低 5.9 GiB，但慢 **89 倍**。

### Muon 为什么慢（这不是实现缺陷，是算法代价）

Muon 对每个 2D 参数做 Newton-Schulz 正交化（5 次迭代矩阵乘）。小模型上可忽略，
但 3B 有 **2550 个参数张量**，逐张量正交化就是 O(张量数 × 迭代数 × 矩阵乘) 的固定开销。
要在大模型上用 Muon，需要按层合并正交化组或降低迭代次数 —— 属于后续工作。

**因此 README 把 AdamW 列为 3B 的默认选择，Muon 标注为"仅显存极度紧张时使用"。**
不把"省了 5.9 GiB"说成优点而不提 89 倍的时间代价。

### 修复：`Fp16MomAdamW.state_bytes()` 把 fp16 动量算成了 5.5 GiB

`m` 与 `v` 是**两个**缓冲：fp16 应是 2×2=4 字节/参数（3B ⇒ 11.1 GiB），
原实现写成 2.0（⇒ 5.5 GiB），会让人误以为还能再多塞一倍。已修正 `bytes_per_param()`。

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
                omv = z.read("tgai_go/train/opt_fp16mv.py").decode("utf-8", "replace")
                checks = {
                    "state_bytes 修正": "return 8.0 if self.fp32_state else 4.0" in omv,
                    "README 有 Muon 慢": "62.9" in body,
                    "README 有 AdamW 709": "709" in body,
                }
                for kk, hit in checks.items():
                    print("   %-20s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
