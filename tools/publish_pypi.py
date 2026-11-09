# -*- coding: utf-8 -*-
"""把 wheel/sdist 发布到 PyPI —— token **保存一次，以后一条命令**。

为什么有这个脚本：旧流程要求每次在 cmd 里 `set TWINE_PASSWORD=...`，所以老忘。
这里把 token 存到用户目录下的私有文件（Windows 上用 icacls 收紧为仅本人可读），
之后直接 `python tools/publish_pypi.py` 即可；token 不会打印、不会进仓库。

用法
    # 第一次（或换 token 时）：交互式粘贴一次，隐藏输入
    python tools/publish_pypi.py --save-token

    # 发布到 TestPyPI 先试
    python tools/publish_pypi.py --test

    # 正式发布
    python tools/publish_pypi.py

    # 只检查某个版本是否已在 PyPI 上（不发包）
    python tools/publish_pypi.py --check

token 读取顺序：环境变量 PYPI_TOKEN → TWINE_PASSWORD → %USERPROFILE%\\.pypi_token
（TestPyPI 用 TESTPYPI_TOKEN / %USERPROFILE%\\.testpypi_token）
"""
import argparse
import getpass
import glob
import io
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")
DEFAULT_DIST = os.path.join(ROOT, "publish_ready")


def _token_path(test):
    return os.path.join(HOME, ".testpypi_token" if test else ".pypi_token")


def _read_token(test):
    for env in (("TESTPYPI_TOKEN",) if test else ("PYPI_TOKEN", "TWINE_PASSWORD")):
        v = os.environ.get(env, "").strip()
        if v:
            return v, "环境变量 %s" % env
    p = _token_path(test)
    if os.path.isfile(p):
        with io.open(p, encoding="utf-8") as f:
            v = f.read().strip()
        if v:
            return v, "文件 %s" % p
    return None, None


def save_token(test):
    p = _token_path(test)
    print("把 PyPI token 粘贴到这里（输入不回显）。创建/轮换：https://pypi.org/manage/account/token/")
    if os.path.isfile(p):
        print("注意：%s 已存在，将被覆盖" % p)
    tok = getpass.getpass("token: ").strip()
    if not tok.startswith("pypi-"):
        print("!! 看起来不是 PyPI token（应以 pypi- 开头）。已取消，未写入。")
        return 2
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(tok + "\n")
    os.chmod(p, 0o600)
    if os.name == "nt":
        # 仅当前用户可读，去掉继承，避免被同机其他账户读到
        subprocess.run(["icacls", p, "/inheritance:r", "/grant:r", '%s:F' % os.environ.get("USERNAME", "")],
                       capture_output=True, text=True)
    print("已保存到 %s（长度 %d，前缀 %s…，仅本人可读）" % (p, len(tok), tok[:8]))
    print("以后发布：python tools/publish_pypi.py")
    return 0


def check_version(version, test=False):
    base = "https://test.pypi.org/pypi" if test else "https://pypi.org/pypi"
    url = "%s/tgai-orv/json" % base
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
        have = sorted(data.get("releases", {}).keys())
        print("PyPI 上已有版本:", have[-6:] if have else "（无）")
        print("目标版本 %s 是否已发布: %s" % (version, version in data.get("releases", {})))
        return 0 if version in data.get("releases", {}) else 1
    except Exception as e:
        print("查询失败（可能需要代理）：%s" % type(e).__name__)
        return 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-token", action="store_true", help="交互式保存 token（隐藏输入）")
    ap.add_argument("--test", action="store_true", help="发到 TestPyPI")
    ap.add_argument("--check", action="store_true", help="只查询版本是否已发布")
    ap.add_argument("--dist", default=DEFAULT_DIST, help="包含 whl/sdist 的目录")
    ap.add_argument("--skip-existing", action="store_true", help="已存在的文件跳过（twine 参数）")
    args = ap.parse_args()

    if args.save_token:
        return save_token(args.test)

    # 读版本
    ver = None
    try:
        with io.open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
            for line in f:
                if line.startswith("version"):
                    ver = line.split("=")[1].strip().strip('"')
                    break
    except Exception:
        pass
    if args.check:
        if not ver:
            print("!! 读不到 pyproject 的版本")
            return 2
        return check_version(ver, args.test)

    files = sorted(glob.glob(os.path.join(args.dist, "*.whl")) +
                   glob.glob(os.path.join(args.dist, "*.tar.gz")))
    if not files:
        print("!! %s 里没有 whl/sdist。先构建：python -m build --outdir dist 并把产物放进该目录" % args.dist)
        return 2
    tok, src = _read_token(args.test)
    if not tok:
        print("!! 没有找到 token。先执行一次：python tools/publish_pypi.py --save-token")
        print("   （也可以设置环境变量 PYPI_TOKEN 后重试）")
        return 2
    print("将要上传 %d 个文件（版本 %s）：" % (len(files), ver))
    for f in files:
        print("   ", os.path.basename(f))
    print("token 来源:", src, "（不打印内容）")

    env = dict(os.environ)
    env.update({"TWINE_USERNAME": "__token__", "TWINE_PASSWORD": tok,
                "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    cmd = [sys.executable, "-m", "twine", "upload"]
    if args.test:
        cmd += ["--repository-url", "https://test.pypi.org/legacy/"]
    if args.skip_existing:
        cmd += ["--skip-existing"]
    cmd += files
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    out = out.replace(tok, "***MASKED***")
    print(out[-2000:])
    print("twine rc=%d" % r.returncode)
    if r.returncode == 0 and ver:
        print("\n发布后校验：")
        check_version(ver, args.test)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
