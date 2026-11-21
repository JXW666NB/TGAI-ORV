# -*- coding: utf-8 -*-
"""词条必须**随 wheel 一起安装**（2026-11-21 发版前发现的问题）。

`wiki/` 在仓库根，而 setuptools 只打 `python/tgai_go/**` ⇒ 从 PyPI 装的用户
拿不到任何词条，`T.help` 的 Wiki 会是空的。修法是保留根目录为作者唯一入口，
用 `scripts/sync_wiki_package.py` 生成包内副本 `python/tgai_go/wiki/`，
再由这个测试保证两边**不漂移**（改了一边没同步 ⇒ CI 红）。
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "wiki")
DST = os.path.join(REPO, "python", "tgai_go", "wiki")


def _entries(root):
    out = {}
    for dirpath, _dirs, names in os.walk(root):
        for fn in names:
            if fn.lower().endswith(".txt"):
                full = os.path.join(dirpath, fn)
                out[os.path.relpath(full, root)] = full
    return out


def test_packaged_wiki_dir_exists():
    assert os.path.isdir(DST), (
        "包内没有 wiki/ 目录 ⇒ PyPI 用户拿不到词条。"
        "修法：运行 python scripts/sync_wiki_package.py")


def test_packaged_copy_matches_repo_root():
    src, dst = _entries(SRC), _entries(DST)
    assert set(src) == set(dst), (
        "包内副本与仓库根词条文件集合不一致：缺 %s / 多 %s。"
        "修法：运行 python scripts/sync_wiki_package.py"
        % (sorted(set(src) - set(dst)), sorted(set(dst) - set(src))))
    for rel in sorted(src):
        with open(src[rel], "rb") as f1, open(dst[rel], "rb") as f2:
            assert f1.read() == f2.read(), (
                "词条 %s 两边内容不同（改了根目录但没同步进包）。"
                "修法：运行 python scripts/sync_wiki_package.py" % rel)


def test_sync_script_reports_consistent():
    r = subprocess.run([sys.executable, "-X", "utf8",
                        os.path.join(REPO, "scripts", "sync_wiki_package.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_declared_in_package_data():
    """`pyproject.toml` 的 package-data 必须包含 wiki/*.txt，否则 wheel 里还是空的。"""
    with open(os.path.join(REPO, "pyproject.toml"), encoding="utf-8") as f:
        txt = f.read()
    assert 'wiki/*.txt' in txt, (
        "pyproject.toml 的 [tool.setuptools.package-data] 没有声明 wiki/*.txt。"
        "修法：给 \"tgai_go\" 加上 \"wiki/*.txt\"")
