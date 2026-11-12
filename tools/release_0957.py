# -*- coding: utf-8 -*-
"""0.9.57：把推理原生库打进 wheel（P0），并让加载失败给出可执行指引。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.57"
ENTRY = """## [0.9.57] - 2026-11-12

**修外部报告的 P0：`pip install` 后推理端直接不可用。**

### 问题

外部实测（GTX 1060 / 干净 venv）：`pip install tgai-orv` 之后 `tgai_go.load()` 报
`FileNotFoundError: DLL not found` —— wheel 里**只有 80 个 py 文件，不含 `_native/`**。
"训推一体"在 pip 形态下是断的。

进一步：把仓库里的 `_native/*.dll` 手工拷进 site-packages 也加载不了，
`ctypes.CDLL` 报 `OSError [WinError 193] 不是有效的 Win32 应用程序`
（两个引擎 DLL 的 PE 签名都比 DOS 头 `e_lfanew` 早 1 字节）。

### 本版改了什么

| 改动 | 说明 |
| --- | --- |
| **`pyproject.toml` 补 package-data** | `"tgai_go" = ["_native/*.dll", "_native/*.so", "_native/*.dylib"]` —— 原生库随 wheel 分发 |
| **`native_available()` 为假时报可执行指引** | 不再抛裸 `FileNotFoundError`；告知升级 / 源码安装 / `dll=` 指定，并说明训练侧不受影响 |
| **原生库加载失败时同理** | 捕获 `OSError`，其中 WinError 193 明确说明"随包 DLL 损坏、需重编"，并给出三条替代路径 |
| **`kernel_count.py` 去硬编码** | 改为读 `V27_RUN` / `V27_TG`，缺失时说明这是作者本机工具并指向 `profile_block.py` |
| **README 数字不再写死** | 版本徽章改为 shields 动态读 PyPI；测试数/内核数指向 `CAPABILITY_AUDIT.md` |
| **仓库卫生** | `backups/`（18 个）与源码内 `.bak`/空 txt 移出索引并 gitignore；`release/` 保留（3 处文档引用）|

### 仍未解决（如实记录）

**随包的 Windows 引擎 DLL 本身是坏的**（PE 头错位 1 字节），需要**重编**才能用；
本机没有 C 编译器，无法在此完成。所以本版把"文件打包对了"这条路修通，
但**推理在 Windows 上仍不可用**，直到 DLL 重编。详见
`docs/EXTERNAL_REVIEW_TRIAGE_2026-11-12.md`。

套件：本地 670 passed / 42 skipped。
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
                nat = [n for n in z.namelist() if "/_native/" in n]
                init = z.read("tgai_go/__init__.py").decode("utf-8", "replace")
                checks = {
                    "_native 随包": bool(nat),
                    "缺库有指引": "找不到推理引擎的原生库" in init,
                    "加载失败有指引": "推理引擎原生库加载失败" in init,
                }
                print("   原生库文件:", [n.split("/_native/")[-1] for n in nat])
                for kk, hit in checks.items():
                    print("   %-16s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
