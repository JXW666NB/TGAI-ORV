# -*- coding: utf-8 -*-
"""发布前自检：把「装上才发现缺东西」这类问题挡在发布之前。

背景（2026-10-30 两次真实事故）：
  1. wheel 里没有 `tgai_go/train/kernels/*.cu` → 用户装上后 lowmem 注意力不可用；
  2. ORV 树缺 `tgai_go/pack.py` 等兄弟模块 → 导出 / 续训时 ModuleNotFoundError。

这两类问题在开发目录里永远测不出来，必须**装到一个干净环境里再 import 一遍**。

用法：
    python tools/check_package.py            # 用 dist/ 里已有的 wheel
    python tools/check_package.py --build    # 先 python -m build
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile
import venv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUST_HAVE_FILES = [
    "tgai_go/__init__.py",
    "tgai_go/pack.py",
    "tgai_go/train/cuda.py",
    "tgai_go/train/__init__.py",
    "tgai_go/train/export.py",
    "tgai_go/train/doctor.py",
    "tgai_go/train/dist.py",
    "tgai_go/train/backend.py",
]
MUST_HAVE_GLOB = ["tgai_go/train/kernels/*.cu"]
IMPORT_CHECKS = [
    "import tgai_go",
    "import tgai_go.pack",
    "import tgai_go.train",
    "import tgai_go.train.export",
    "import tgai_go.train.doctor",
    "import tgai_go.train.dist",
    "import tgai_go.train.backend",
    "from tgai_go.train.cuda import attn_nop_source_path",
]


def sh(cmd, **kw):
    r = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace",
                       capture_output=True, **kw)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    bad = []

    if a.build:
        rc, out = sh([sys.executable, "-m", "build"])
        print("[build] %s" % ("OK" if rc == 0 else "FAIL"))
        if rc:
            print(out[-800:])
            return 1

    wheels = sorted(glob.glob(os.path.join("dist", "*.whl")))
    if not wheels:
        print("找不到 dist/*.whl，先跑 python -m build 或加 --build")
        return 1
    wheel = wheels[-1]
    print("检查 wheel：%s" % os.path.basename(wheel))

    # 1) 文件清单
    import zipfile
    names = set(zipfile.ZipFile(wheel).namelist())
    for f in MUST_HAVE_FILES:
        ok = f in names
        print("  %-42s %s" % (f, "OK" if ok else "缺失"))
        if not ok:
            bad.append(f)
    for pat in MUST_HAVE_GLOB:
        hit = [n for n in names if glob.fnmatch.fnmatch(n, pat)]
        print("  %-42s %s" % (pat, "OK(%d)" % len(hit) if hit else "缺失"))
        if not hit:
            bad.append(pat)

    # 2) wheel 里的 .py 全部能编译（挡住缩进/语法错误）
    for n in sorted(names):
        if n.endswith(".py"):
            try:
                compile(zipfile.ZipFile(wheel).read(n).decode("utf-8-sig"), n, "exec")
            except SyntaxError as exc:
                print("  语法错误 %s: %s" % (n, exc))
                bad.append(n)

    # 3) 装进干净 venv 再 import（用系统包避免下载 numpy）
    tmp = tempfile.mkdtemp(prefix="orv_pkgchk_")
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(tmp)
    py = os.path.join(tmp, "Scripts" if os.name == "nt" else "bin",
                      "python.exe" if os.name == "nt" else "python")
    rc, out = sh([py, "-m", "pip", "install", "--no-deps", "-q", wheel])
    print("[install] %s" % ("OK" if rc == 0 else "FAIL"))
    if rc:
        print(out[-600:])
        bad.append("install")

    code = "; ".join(IMPORT_CHECKS) + "; print('IMPORTS_OK')"
    rc, out = sh([py, "-c", code], cwd=tempfile.gettempdir())
    print("[imports] %s" % ("OK" if "IMPORTS_OK" in out else "FAIL"))
    if "IMPORTS_OK" not in out:
        print(out[-800:])
        bad.append("imports")

    # 4) 算子源码能被定位
    rc, out = sh([py, "-c",
                  "from tgai_go.train.cuda import attn_nop_source_path as f;"
                  "import os; p=f(); print('CU_OK', os.path.getsize(p))"],
                 cwd=tempfile.gettempdir())
    print("[cu source] %s" % ("OK" if "CU_OK" in out else "FAIL"))
    if "CU_OK" not in out:
        print(out[-600:])
        bad.append("cu")

    print("")
    if bad:
        print("结论：FAIL（%d 项）—— %s" % (len(bad), ", ".join(map(str, bad[:6]))))
        return 1
    print("结论：PASS —— 这个 wheel 可以被安全地发到 PyPI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
