# -*- coding: utf-8 -*-
"""文件卫生：包树与文档/工具目录必须 UTF-8 无 BOM、LF 行尾。

为什么要有这个测试：工程标准里写了「提交前无 BOM、无 CRLF」，但之前**没有任何自动检查**，
结果包树里 16 个文件带着 CRLF 被打进了 wheel（用户拿到的是混合行尾的源码）。
规则不自动化 = 靠自觉 = 迟早破坏。这个测试把那条规则变成 CI 会红的断言。

范围：
  * python/tgai_go/**  —— 会被打进 wheel 的东西（最重要）
  * tools/ tests/ docs/ —— 仓库里给人看/给人用的东西

不在范围（已记录的欠账）：仓库根目录的历史脚本（tgai*.py 等）当年是 CRLF，
一次性全改会把 259 个文件的 diff 混进来；需要时单独提交。
"""
import io
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREES = ("python/tgai_go", "tools", "tests", "docs")
EXTS = (".py", ".md", ".toml", ".yml", ".yaml", ".c", ".h", ".cu", ".json", ".txt", ".sh")
SKIP_DIRS = {".git", "build", "dist", "__pycache__", ".pytest_cache", ".eggs"}


def _files():
    for tree in TREES:
        top = os.path.join(REPO, tree)
        if not os.path.isdir(top):
            continue
        for root, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs
                       if d not in SKIP_DIRS and not d.endswith(".egg-info")]
            for fn in files:
                if fn.lower().endswith(EXTS):
                    yield os.path.join(root, fn)


def test_no_bom():
    bad = []
    for p in _files():
        with open(p, "rb") as f:
            if f.read(3) == b"\xef\xbb\xbf":
                bad.append(os.path.relpath(p, REPO))
    assert not bad, "这些文件带 UTF-8 BOM（会让 Python 报 invalid non-printable character）：%s" % bad


def test_no_crlf():
    bad = []
    for p in _files():
        with open(p, "rb") as f:
            if b"\r\n" in f.read():
                bad.append(os.path.relpath(p, REPO))
    assert not bad, ("这些文件是 CRLF 行尾（打包后用户会拿到混合行尾）：%s" % bad[:10]
                     + ("" if len(bad) <= 10 else " … 共 %d 个" % len(bad)))


def test_package_files_are_utf8():
    """包树里的 .py 必须能被 UTF-8 解码（我们全中文注释，坏一个就 import 失败）。"""
    for root, dirs, files in os.walk(os.path.join(REPO, "python", "tgai_go")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            with open(p, "rb") as f:
                b = f.read()
            try:
                b.decode("utf-8")
            except UnicodeDecodeError as ex:        # noqa: PERF203
                pytest.fail("%s 不是合法 UTF-8：%s" % (os.path.relpath(p, REPO), ex))
