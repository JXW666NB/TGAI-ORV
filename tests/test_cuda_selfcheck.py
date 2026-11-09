# -*- coding: utf-8 -*-
"""静态自检：`cuda.py` 里每个 `self.xxx(...)` 调用都必须有对应的方法定义。

为什么需要它：曾经出现过 `log_softmax_lastdim` 调用 `self._warp_row_grid(...)`，
但整个文件里**没有这个方法的定义** —— CPU 测试走 numpy 路径、V27 训练走 `softmax`，
所以一直没被触发，直到在 GPU 上第一次用 `log_softmax` 才炸。
这类"缺方法"的 bug 完全可以静态查出来，不需要 GPU。
"""
import ast
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA = os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py")


def _classes(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield node


def test_every_self_call_has_a_definition():
    src = io.open(CUDA, encoding="utf-8").read()
    tree = ast.parse(src)
    missing = []
    for cls in _classes(tree):
        defined = set()
        assigned = set()
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        assigned.add(tgt.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                assigned.add(node.target.id)
        for node in ast.walk(cls):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "self":
                name = fn.attr
                if name not in defined and name not in assigned:
                    missing.append("%s.%s (line %d)" % (cls.name, name, node.lineno))
    assert not missing, "这些 self 调用没有定义：" + "; ".join(sorted(set(missing)))


def test_no_duplicate_method_names_in_runtime_class():
    """同名方法重复定义会静默覆盖（后者生效），这里显式报出来。"""
    tree = ast.parse(io.open(CUDA, encoding="utf-8").read())
    dup = []
    for cls in _classes(tree):
        seen = {}
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in seen:
                    dup.append("%s.%s (line %d 与 %d)" % (cls.name, node.name, seen[node.name], node.lineno))
                seen[node.name] = node.lineno
    assert not dup, "重复定义：" + "; ".join(dup)
