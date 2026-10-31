# -*- coding: utf-8 -*-
"""版本一致性：pyproject.toml 与 tgai_go.__init__ 的退回字面量必须相同。

为什么要有：`__version__` 曾经是写死的 `1.4.1`，而 `pyproject.toml` 是 `0.2.0` ——
`pip show tgai-orv` 与 `python -m tgai_go --version` 报两个不同的版本号，
用户报 bug 时根本对不上。现在版本只有一个来源（安装元数据优先），
退回字面量由这个测试守着，任何一次改版本漏了都会红。
"""
import io
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with io.open(os.path.join(REPO, *parts), encoding="utf-8") as f:
        return f.read()


def test_fallback_version_matches_pyproject():
    py = re.search(r'(?m)^version\s*=\s*"([^"]+)"', _read("pyproject.toml"))
    assert py, "pyproject.toml 里找不到 version"
    init = _read("python", "tgai_go", "__init__.py")
    # 取文件里所有 __version__ = "x" 的字面量，最后一个就是退回值
    lits = re.findall(r'(?m)^\s*__version__\s*=\s*"([^"]+)"', init)
    assert lits, "tgai_go/__init__.py 里找不到 __version__ 字面量"
    assert lits[-1] == py.group(1), (
        "pyproject 是 %s，退回字面量是 %s —— 改版本时漏了一处" % (py.group(1), lits[-1]))


def test_runtime_version_is_pep440():
    import sys
    sys.path.insert(0, os.path.join(REPO, "python"))
    import tgai_go
    assert re.match(r"^\d+\.\d+\.\d+", tgai_go.__version__), tgai_go.__version__


def test_changelog_has_current_version_section():
    py = re.search(r'(?m)^version\s*=\s*"([^"]+)"', _read("pyproject.toml")).group(1)
    cl = _read("CHANGELOG.md")
    assert ("## [%s]" % py) in cl, "CHANGELOG 里没有当前版本 %s 的段落" % py
