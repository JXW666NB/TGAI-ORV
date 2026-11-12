# -*- coding: utf-8 -*-
"""0.9.55：Pascal 共享内存守卫（外部报告 P1）+ 问题台账入库。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.55"
ENTRY = """## [0.9.55] - 2026-11-12

**修外部实测的 P1：Pascal 消费卡上融合注意力反向报错看不懂。**

### 问题（外部实测，GTX 1060 / sm_61）

`scaled_dot_product_attention(..., lowmem=True)` 前向正常，**反向抛**
`cuFuncSetAttribute(attn_bwd_flash maxdyn) rc=1 invalid argument` —— 完全看不出原因。

真因：融合注意力反向要 `_ATTN_FLASH_SMEM = 68 KB`（g2 变体 84 KB），
而 sm_61 单 block 动态 smem 上限只有 **48 KB 且不支持 opt-in**；
而 opt-in 的设置调用排在检查之前，于是先炸在那里。

### 改法

* 新增 `_device_smem_limit()`：先取 `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`（属性 97），
  取不到退回 `MAX_SHARED_MEMORY_PER_BLOCK`（属性 8）；
* `_attn_flash_launch` **把上限检查提到 opt-in 之前**，超限时报：

```
融合注意力反向需要 68 KB 动态共享内存，本卡（sm_61）上限只有 48 KB。
修法：改用 lowmem=False（默认，走物化注意力，多花显存但能出结果）；或减小 seq / batch。
Pascal 消费卡（sm_61 等）上限 48 KB 且不支持 opt-in，该路径在那些卡上不可用；
V100（sm_70）可 opt-in 到 96 KB 正常使用
```

### 验证

* 新增 `tests/test_attn_smem_guard.py`（2 例）：模拟本卡上限 48 KB，断言报错含上限、`修法`、
  `lowmem=False` 与卡型提示；另验上限足够时**不误伤**且叶子张量真拿到梯度。**V100 上 2 passed**。
* `tools/attn_ref.py --check`：dh=64 的 16 个数组与改前**最大绝对差 0.000e+00** ⇒ 正常路径未受影响。

### 文档

新增 `docs/EXTERNAL_REVIEW_TRIAGE_2026-11-12.md`：外部报告 8 条问题的台账，
逐条标明「已修 / 未修」。**未修的两条 P0（wheel 缺 `_native`、仓库 DLL 的 PE 头错位 1 字节）
明确写清，推理端在 pip 形态下不可用是已知状态。**

套件：本地（无 GPU）**670 passed / 42 skipped**；V100 上 **672 passed / 40 skipped**
（新增的 2 例在无 CUDA 时跳过）。
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
                core = z.read("tgai_go/train/cuda.py").decode("utf-8", "replace")
                checks = {
                    "有上限查询": "_device_smem_limit" in core,
                    "守卫在 opt-in 前": core.find("_device_smem_limit()")
                                       < core.find("_attn_set_dyn_smem(name, _ATTN_FLASH_MAX_SMEM)"),
                    "报错含 Pascal": "Pascal 消费卡" in core,
                }
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
