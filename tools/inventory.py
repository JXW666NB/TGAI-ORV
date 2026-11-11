# -*- coding: utf-8 -*-
"""能力盘点：扫代码与测试，产出可审计的清单。

## 为什么用脚本而不是手写

社区要照着这份清单干活，手写必然漏、必然过期。
这里从**源码与测试**直接提取，每个条目都能追到出处。

输出：docs/CAPABILITY_AUDIT.md
"""
import ast
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "python", "tgai_go", "train")


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def module_inventory():
    """每个模块的公开类与函数。"""
    out = {}
    for fn in sorted(os.listdir(PKG)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        p = os.path.join(PKG, fn)
        try:
            tree = ast.parse(read(p))
        except SyntaxError:
            out[fn] = ("<语法错误>", [], [])
            continue
        cls, fns = [], []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                cls.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and not node.name.startswith("_"):
                fns.append(node.name)
        out[fn] = (len(read(p)) // 1024, cls, fns)
    return out


def cuda_ops():
    """cuda.py 暴露的 rt.* 方法与内核数。"""
    s = read(os.path.join(PKG, "cuda.py"))
    methods = sorted(set(re.findall(r"(?m)^    def (\w+)\(", s)) - {"__init__"})
    kernels = sorted(set(re.findall(r'__global__ void (\w+)\(', s)))
    return methods, kernels, len(s) // 1024


def nn_layers():
    """nn 命名空间里挂了什么。"""
    s = read(os.path.join(PKG, "__init__.py"))
    i = s.find("class _NNModule")
    seg = s[i:i + 2600] if i > 0 else ""
    names = re.findall(r"(?m)^    (\w+) = ", seg)
    return names


def optimizers():
    s = read(os.path.join(PKG, "__init__.py"))
    i = s.find("class _OptimModule")
    seg = s[i:i + 900] if i > 0 else ""
    return re.findall(r"(?m)^    (\w+) = ", seg)


def tests():
    d = os.path.join(ROOT, "tests")
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".py") and fn.startswith("test_"):
            body = read(os.path.join(d, fn))
            n = len(re.findall(r"(?m)^def test_", body))
            out.append((fn, n))
    return out


def main():
    mods = module_inventory()
    methods, kernels, cuda_kb = cuda_ops()
    layers = nn_layers()
    opts = optimizers()
    ts = tests()

    L = []
    L.append("# ORV 能力盘点（脚本生成，可复核）")
    L.append("")
    L.append("> 由 `tools/inventory.py` 从**源码与测试**直接提取，不是手写记忆。")
    L.append("> 重新生成：`python tools/inventory.py`")
    L.append("")
    L.append("## 汇总")
    L.append("")
    L.append("| 项 | 数量 |")
    L.append("| --- | --- |")
    L.append("| Python 模块（不含下划线开头） | %d |" % len(mods))
    L.append("| 后端接口方法（`rt.*`） | %d |" % len(methods))
    L.append("| CUDA 内核 | %d |" % len(kernels))
    L.append("| `nn` 命名空间导出 | %d |" % len(layers))
    L.append("| 核心优化器 | %d |" % len(opts))
    L.append("| 测试文件 | %d |" % len(ts))
    L.append("| 测试用例 | %d |" % sum(n for _, n in ts))
    L.append("| `cuda.py` 规模 | %d KB |" % cuda_kb)
    L.append("")

    L.append("## 模块清单")
    L.append("")
    L.append("| 模块 | KB | 公开类 | 公开函数 |")
    L.append("| --- | --- | --- | --- |")
    for fn, (kb, cls, fns) in mods.items():
        L.append("| `%s` | %s | %s | %s |" % (
            fn, kb,
            ", ".join("`%s`" % c for c in cls[:6]) or "—",
            ", ".join("`%s`" % f for f in fns[:8]) or "—"))
    L.append("")

    L.append("## `nn` 命名空间")
    L.append("")
    L.append(" ".join("`%s`" % x for x in layers))
    L.append("")
    L.append("## 优化器")
    L.append("")
    L.append(" ".join("`%s`" % x for x in opts))
    L.append("")
    L.append("## CUDA 内核（%d 个）" % len(kernels))
    L.append("")
    L.append(" ".join("`%s`" % k for k in kernels))
    L.append("")
    L.append("## 后端接口方法（%d 个）" % len(methods))
    L.append("")
    L.append(" ".join("`%s`" % m for m in methods))
    L.append("")
    L.append("## 测试文件与用例数")
    L.append("")
    L.append("| 文件 | 用例 |")
    L.append("| --- | --- |")
    for fn, n in ts:
        L.append("| `%s` | %d |" % (fn, n))
    L.append("")

    p = os.path.join(ROOT, "docs", "CAPABILITY_AUDIT.md")
    io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(L))
    print("已生成 docs/CAPABILITY_AUDIT.md")
    print("模块 %d / 后端方法 %d / 内核 %d / nn %d / 优化器 %d / 测试用例 %d"
          % (len(mods), len(methods), len(kernels), len(layers), len(opts),
             sum(n for _, n in ts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
