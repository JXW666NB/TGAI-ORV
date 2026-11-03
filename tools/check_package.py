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


def _clean_stale_wheels():
    """删掉 dist/ 里本项目旧 wheel。

    为什么必须删：`--build` 之后我们只检查 `dist/*.whl` 里**最新**的一个，
    但只要目录里还躺着旧包，就存在"检查错了对象"的风险（曾经就是这样偶发 FAIL 的）。
    不直接删用户的东西 —— 挪到 `dist/_stale_<时间戳>/`。
    """
    old = [w for w in glob.glob(os.path.join("dist", "*.whl"))
           if os.path.basename(w).startswith("tgai_orv")]
    if not old:
        return
    import time as _t
    box = os.path.join("dist", "_stale_%s" % _t.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(box, exist_ok=True)
    for w in old:
        try:
            os.replace(w, os.path.join(box, os.path.basename(w)))
        except OSError:
            pass
    print("[build] 旧 wheel %d 个已挪到 %s" % (len(old), box))


def _acquire_build_lock():
    """并发构建会让两次 build 写同一个 dist/ 而互相踩，这里用带过期保护的锁。"""
    import time as _t
    lock = os.path.join("dist", ".build.lock")
    if os.path.isfile(lock):
        age = _t.time() - os.path.getmtime(lock)
        if age > 1800:                     # 30 分钟视为残留（上次被中断）
            print("[build] 清理残留构建锁（已 %.0f 秒）" % age)
            try:
                os.remove(lock)
            except OSError:
                pass
        else:
            print("另一个构建正在进行（dist/.build.lock，%.0f 秒）。"
                  "修法：等它结束；若确认没有在跑，删掉该文件再重试" % age)
            return None
    os.makedirs("dist", exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("抢锁失败（另一个构建刚起来）。修法：稍后重试")
        return None
    os.write(fd, str(os.getpid()).encode())
    return (lock, fd)


def _release_build_lock(handle):
    if not handle:
        return
    lock, fd = handle
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        os.remove(lock)
    except OSError:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    bad = []

    if a.build:
        lock = _acquire_build_lock()
        if lock is None:
            return 1
        try:
            _clean_stale_wheels()
            rc, out = sh([sys.executable, "-m", "build"])
        finally:
            _release_build_lock(lock)
        print("[build] %s" % ("OK" if rc == 0 else "FAIL"))
        if rc:
            print(out[-800:])
            return 1

    wheels = sorted(glob.glob(os.path.join("dist", "*.whl")))
    if not wheels:
        print("找不到 dist/*.whl，先跑 python -m build 或加 --build")
        return 1
    wheel = wheels[-1]
    # 透明化：打印路径/时间/大小。**残留的旧 wheel 会被当成新包自检**，
    # 以前就是"打包自检偶发 FAIL"的来源之一（并发构建 + dist/ 撞车）。
    st = os.stat(wheel)
    import time as _t
    print("检查 wheel：%s" % os.path.basename(wheel))
    print("           %s  已在 dist/ 里躺了 %.0f 秒（%d 字节）"
          % (_t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(st.st_mtime)),
             _t.time() - st.st_mtime, st.st_size))
    if _t.time() - st.st_mtime > 3600:
        print("  ⚠ 这个 wheel 超过 1 小时了；确认它就是你刚构建的那份（否则先删 dist/）")

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

    # 2) wheel 里的 .py 全部能编译（挡住缩进/语法错误），
    #    并检查**用户拿到的东西**本身没有 BOM / CRLF。
    #    这两条以前只写在文档里、没有自动检查，结果包树里 16 个文件带着 CRLF 发布出去了。
    zf = zipfile.ZipFile(wheel)
    bom_hit, crlf_hit = [], []
    for n in sorted(names):
        if not (n.endswith(".py") or n.endswith(".md") or n.endswith(".txt")
                or n.endswith(".cu") or n.endswith(".h")):
            continue
        raw = zf.read(n)
        if raw.startswith(b"\xef\xbb\xbf"):
            bom_hit.append(n)
        if b"\r\n" in raw:
            crlf_hit.append(n)
        if n.endswith(".py"):
            try:
                compile(raw.decode("utf-8-sig"), n, "exec")
            except SyntaxError as exc:
                print("  语法错误 %s: %s" % (n, exc))
                bad.append(n)
    print("  %-42s %s" % ("wheel 内无 BOM", "OK" if not bom_hit else "FAIL %s" % bom_hit[:3]))
    print("  %-42s %s" % ("wheel 内无 CRLF", "OK" if not crlf_hit else "FAIL %s" % crlf_hit[:3]))
    if bom_hit:
        bad.extend(bom_hit)
    if crlf_hit:
        bad.extend(crlf_hit)

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
