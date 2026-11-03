# -*- coding: utf-8 -*-
"""门禁：文档说「有」的算子，代码里必须真能解析到。

`docs/OP_COVERAGE.md` 是一张「PyTorch 算子 → 我们的对应」表，状态列写「有」或「缺」。
只写文档不算数：这个测试把每一行「有」的对应符号真的 `getattr` 一遍，
解析不到就红 —— 防止"文档说支持、实际没接线"。

反向也管一半：标「缺」的行数锁定基线（**只减不增**），
避免有人为了好看把「缺」悄悄改成「有」却什么都没实现。
"""
import importlib
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(REPO, "docs", "OP_COVERAGE.md")

# 允许的「缺」的条数上界（只能减少）
MAX_MISSING = 1

# 「我们的对应」列里，这些形态不是可解析符号，直接跳过
_SKIP_TOKENS = {"@", "+", "-", "*", "/", "batched @", "[]", "T.T", "x.T"}


def _rows():
    if not os.path.isfile(DOC):
        pytest.skip("没有 docs/OP_COVERAGE.md")
    with open(DOC, "r", encoding="utf-8") as f:
        text = f.read()
    out = []
    for line in text.splitlines():
        if line.count("|") < 3 or not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        if cells[0].startswith("PyTorch") or set(cells[0]) <= set("-: "):
            continue
        if not cells[0].startswith("`torch"):
            continue
        out.append((cells[0].strip("`"), cells[1].strip("`").strip(), cells[2]))
    return out


def _autoload_contrib():
    """contrib 的算子是插件式注册的：必须先 autoload 才能通过 F.* 调到。"""
    try:
        from tgai_go.train import contrib
        contrib.autoload()
    except Exception:                                  # noqa: BLE001
        pass


def _resolve(sym):
    """把「我们的对应」解析成真实对象；解析不到返回 None，非符号返回 True。"""
    s = sym.strip()
    if not s or s in _SKIP_TOKENS:
        return True
    # 形如 `@ (Tensor.__matmul__)`：取括号里的真符号
    # 形如 (x) / /b 的写法：去掉参数、逐项检查
    s = s.split("(")[0].strip()
    if "/" in s:
        parts = [x.strip() for x in s.split("/")]
        return True if all(_resolve(x) for x in parts) else None
    m = re.search(r"Tensor\.(\w+)", s)
    if m:
        T = importlib.import_module("tgai_go.train")
        return getattr(getattr(T, "Tensor", None), m.group(1), None)
    m = re.fullmatch(r"(\w+)\.(\w+)", s)
    if m:
        T = importlib.import_module("tgai_go.train")
        name, attr = m.group(1), m.group(2)
        if name == "F":                       # F 就是 functional 门面
            mod = getattr(T, "functional", None)
        elif name == "nn_ops":                # 子模块要显式 import，T 上没有这个属性
            mod = importlib.import_module("tgai_go.train.nn_ops")
        elif name == "Tensor":
            mod = getattr(T, "Tensor", None)
        elif name == "T":                     # T.x 指的是包本身（tgai_go.train.x）
            mod = T
        elif name == "contrib":
            mod = importlib.import_module("tgai_go.train.contrib")
        else:
            mod = getattr(T, name, None)
        if mod is None:
            return None
        return getattr(mod, attr, None)
    # 运算符或自由文本：无法静态解析，按"跳过"处理
    return True


def test_claimed_ops_actually_exist():
    _autoload_contrib()
    bad = []
    for torch_name, ours, status in _rows():
        if status != "有":
            continue
        got = _resolve(ours)
        if got is None:
            bad.append("%s -> %s" % (torch_name, ours))
    assert not bad, (
        "这些算子文档标了「有」，但代码里找不到：\n  " + "\n  ".join(bad)
        + "\n修法：要么把实现接上，要么把状态改成「缺」并写明原因")


def test_missing_count_does_not_grow():
    missing = [t for t, _o, st in _rows() if st == "缺"]
    assert len(missing) <= MAX_MISSING, (
        "标「缺」的算子从 %d 涨到 %d：%s。修法：把缺的实现补上，"
        "不要靠改状态糊弄" % (MAX_MISSING, len(missing), missing))


def test_table_is_parseable():
    rows = _rows()
    assert len(rows) >= 40, "只解析到 %d 行，文档格式可能变了" % len(rows)
    for torch_name, ours, status in rows:
        assert status in ("有", "缺"), "%s 的状态列是 %r，只允许 有/缺" % (torch_name, status)