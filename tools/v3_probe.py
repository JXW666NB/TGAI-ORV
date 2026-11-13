# -*- coding: utf-8 -*-
"""V3 架构体检：类清单、配置字段、每层用到的算子、参数量。

用于评估"把 V3 移植到 ORV"的真实工作量 —— 先知道它用了哪些算子，
再对照 ORV 的算子覆盖面看缺口。

用法：`python v3_probe.py <model_v3.py>`
"""
import ast
import json
import sys


def main():
    path = sys.argv[1]
    src = open(path, encoding="utf-8", errors="replace").read()
    tree = ast.parse(src)
    out = {"file": path, "bytes": len(src)}
    classes, ops = [], set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            classes.append({"name": node.name,
                            "bases": [ast.unparse(b) for b in node.bases][:3],
                            "n_methods": len(methods),
                            "methods": methods[:12]})
        elif isinstance(node, ast.FunctionDef):
            classes.append({"name": node.name, "kind": "function"})
    # 收集用到的 torch 算子（属性访问链）
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            base = node
            parts = []
            while isinstance(base, ast.Attribute):
                parts.append(base.attr)
                base = base.value
            if isinstance(base, ast.Name) and base.id in ("torch", "F", "nn"):
                parts.append(base.id)
                ops.add(".".join(reversed(parts)))
    out["classes"] = classes
    out["torch_ops"] = sorted(ops)
    out["n_torch_ops"] = len(ops)
    # dataclass 字段
    cfg = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and "Config" in node.name:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    cfg[stmt.target.id] = ast.unparse(stmt.annotation)
    out["config_class_fields"] = cfg
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
