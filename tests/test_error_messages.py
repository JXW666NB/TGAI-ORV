# -*- coding: utf-8 -*-
"""门禁：错误信息必须写清「怎么修」。

目标原文：*错误信息：每条都写清「哪个算子 / 什么形状 / 怎么修」*。
只写进文档不算数，这里用 AST 扫一遍源码里的 `raise`，要求消息里出现「修法」。

## 适用范围（逐步扩大）

* 本会话新增/改写的模块：`ops_*`、`dtypes`、`dist_p2p`、`graph_train` —— **强制**。
* 历史模块（`backend.py` / `cuda.py` / `__init__.py` 等）量大，先只做**统计**不拦，
  免得一次改动太大；等逐个模块重写时再纳入（见文件末尾 `LEGACY` 列表）。

## 动态消息怎么处理

`raise ValueError(msg)` 里 `msg` 是变量时无法静态判断，**跳过并计数**，
不假装通过、也不误杀。
"""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO, "python", "tgai_go", "train")

ENFORCED = [
    "ops_cuda.py", "ops_cpu.py", "ops_vk.py", "ops_vk_compute.py", "ops_vk_ops.py",
    "ops_bundle.py", "ops_native.py", "dtypes.py", "dist_p2p.py", "graph_train.py",
]

LEGACY = [
    "backend.py", "cuda.py", "__init__.py", "qwen.py", "checkpoints.py", "data.py",
    "bpe.py", "dist.py", "doctor.py", "export.py", "nn_ops.py", "pipeline.py", "amp.py",
]


def _scan(path):
    """返回 (缺修法的 [(行号, 类型)], 动态消息条数, 总数)。"""
    src = open(path, "r", encoding="utf-8").read()
    tree = ast.parse(src, filename=path)
    bad, dynamic, total = [], 0, 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        total += 1
        if not isinstance(node.exc, ast.Call) or not node.exc.args:
            dynamic += 1
            continue
        seg = ast.get_source_segment(src, node) or ""
        if "修法" in seg:
            continue
        arg0 = node.exc.args[0]
        if isinstance(arg0, ast.JoinedStr) and "修法" in seg:
            continue
        if isinstance(arg0, (ast.Name, ast.Attribute, ast.Subscript)):
            dynamic += 1          # 消息是变量，静态判不了
            continue
        bad.append((node.lineno, getattr(node.exc.func, "id", "?")))
    return bad, dynamic, total


def test_new_modules_have_fix_hints():
    problems = []
    for name in ENFORCED:
        path = os.path.join(TRAIN, name)
        if not os.path.isfile(path):
            continue
        bad, _dyn, _total = _scan(path)
        for line, kind in bad:
            problems.append("%s:%d (%s)" % (name, line, kind))
    assert not problems, (
        "这些 raise 的消息里没有「修法」，用户看到会不知道怎么处理：\n  "
        + "\n  ".join(problems)
        + "\n修法：在消息末尾加一句「修法：…」，写清该改什么/该传什么/该装什么")


def test_legacy_modules_are_tracked():
    """历史模块只统计，不拦；但**必须能被统计到**（防止文件改名后悄悄漏掉）。"""
    seen = 0
    for name in LEGACY:
        path = os.path.join(TRAIN, name)
        if os.path.isfile(path):
            seen += 1
    assert seen >= 8, "历史模块清单与实际文件差异过大，请更新 LEGACY 列表"


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
