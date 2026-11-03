# -*- coding: utf-8 -*-
"""门禁：错误信息必须写清「怎么修」。

目标原文：*错误信息：每条都写清「哪个算子 / 什么形状 / 怎么修」*。
只写进文档不算数 —— 这里用 AST 扫源码，要求 `raise` 的消息里出现「修法」。

## 两级策略

1. **强制（全绿）**：除 `cuda.py` / `__init__.py` 外的所有模块，缺「修法」= 0 条。
2. **棘轮（只减不增）**：`cuda.py`(149) 与 `__init__.py`(119) 存量太大，
   先锁定基线：**新增的 raise 必须带修法，否则这条测试红**；
   以后逐步补完再把基线往下调（改 `BASELINE` 即可）。

## 动态消息怎么处理

`raise ValueError(msg)` 里 `msg` 是变量时静态判不了 —— **跳过并计数**，
不假装通过、也不误杀。`raise SystemExit(main())` 属控制流（CLI 退出码），不算错误信息。
"""
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO, "python", "tgai_go", "train")

# 存量基线：只允许减少，不允许增加（增加会让测试红）
BASELINE = {
    "cuda.py": 149,
    "__init__.py": 119,
}


def _scan(path):
    """返回 (缺修法的 [行号], 动态消息条数, 有效 raise 总数)。"""
    src = open(path, "r", encoding="utf-8").read()
    tree = ast.parse(src, filename=path)
    bad, dynamic, total = [], 0, 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        if not isinstance(node.exc, ast.Call) or not node.exc.args:
            dynamic += 1
            continue
        fn = getattr(node.exc.func, "id", "")
        if fn in ("SystemExit", "KeyboardInterrupt"):
            continue
        total += 1
        seg = ast.get_source_segment(src, node) or ""
        if "修法" in seg:
            continue
        if isinstance(node.exc.args[0], (ast.Name, ast.Attribute, ast.Subscript)):
            dynamic += 1
            continue
        bad.append(node.lineno)
    return bad, dynamic, total


def _modules():
    return sorted(m for m in os.listdir(TRAIN) if m.endswith(".py"))


def test_enforced_modules_have_zero_missing():
    problems = []
    for name in _modules():
        if name in BASELINE:
            continue
        bad, _dyn, _tot = _scan(os.path.join(TRAIN, name))
        for line in bad:
            problems.append("%s:%d" % (name, line))
    assert not problems, (
        "这些 raise 的消息里没有「修法」，用户会不知道怎么处理：\n  "
        + "\n  ".join(problems)
        + "\n修法：在消息末尾加一句「修法：…」，写清该改什么/该传什么/该装什么")


def test_baseline_modules_do_not_regress():
    """棘轮：cuda.py / __init__.py 的缺修法条数只能减少，不能增加。"""
    grew = []
    for name, allowed in BASELINE.items():
        path = os.path.join(TRAIN, name)
        if not os.path.isfile(path):
            grew.append("%s 不存在了（请更新 BASELINE）" % name)
            continue
        bad, _dyn, _tot = _scan(path)
        if len(bad) > allowed:
            grew.append("%s 从 %d 涨到 %d（新增的 raise 必须带「修法」）"
                        % (name, allowed, len(bad)))
    assert not grew, "\n".join(grew)


def test_scanner_itself_works(tmp_path):
    """扫描器自测：明显缺修法的 raise 必须被抓出来。"""
    p = tmp_path / "sample.py"
    p.write_text("def f():\n    raise ValueError('坏了')\n", encoding="utf-8")
    bad, dyn, total = _scan(str(p))
    assert total == 1 and dyn == 0 and len(bad) == 1

    p2 = tmp_path / "ok.py"
    p2.write_text("def f():\n    raise ValueError('坏了。修法：改参数')\n", encoding="utf-8")
    bad2, _d, _t = _scan(str(p2))
    assert not bad2

    p3 = tmp_path / "ctrl.py"
    p3.write_text("def f():\n    raise SystemExit(2)\n", encoding="utf-8")
    _b, _dyn3, total3 = _scan(str(p3))
    assert total3 == 0, "SystemExit 属控制流，不该计入错误信息"