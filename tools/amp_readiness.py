# -*- coding: utf-8 -*-
"""真 AMP 的差距清单：哪些内核必须支持 bf16 才能把激活存成 bf16。

为什么需要这个工具：L3 有一条「真 AMP（激活 bf16 存储）」。如果不量化，
它永远是一句口号；量化之后才知道它是"改 15 个内核"还是"重写半个引擎"。

现状（本工具扫出来的事实）：
  * matmul 走 cuBLAS，已经能用 CUDA_R_16BF（`TG_BF16=1` 打开），这部分**不缺**；
  * 但 `cuda.py` 里的手写内核**全部只接受 float**，所以要"激活存 bf16"，
    得给逐元素/归一化/注意力/交叉熵这些内核加 bf16 变体。

用法：
    python tools/amp_readiness.py            # 打印清单与统计
    python tools/amp_readiness.py --json      # 机器可读
"""
import argparse
import io
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA = os.path.join(REPO, "python", "tgai_go", "train", "cuda.py")

#: 按用途分类：AMP 第一步只需要"张量进出"的内核支持 bf16
CATEGORIES = [
    ("逐元素（加减乘/激活）", r"ewise|scale|silu|gelu|add|mul|fill|copy|cast"),
    ("归一化", r"norm|rms|layernorm"),
    ("注意力 / softmax", r"attn|softmax|rope|mask"),
    ("交叉熵 / 损失", r"ce_|loss|nll"),
    ("嵌入 gather/scatter", r"emb"),
    ("矩阵乘 / 转置", r"gemm|matmul|mm_|transpose"),
    ("归约 / 其它", r"."),
]


def scan():
    s = io.open(CUDA, encoding="utf-8").read()
    sigs = re.findall(r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)", s)
    out = []
    for name, args in sigs:
        a = " ".join(args.split())
        has_bf16 = ("bfloat16" in a) or ("__nv_bfloat16" in a)
        out.append({"name": name, "bf16": has_bf16, "args": a[:120]})
    return out


def classify(name):
    for label, pat in CATEGORIES:
        if re.search(pat, name, re.I):
            return label
    return "归约 / 其它"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    ks = scan()
    by_cat = {}
    for k in ks:
        cat = classify(k["name"])
        d = by_cat.setdefault(cat, {"total": 0, "bf16": 0, "float_only": []})
        d["total"] += 1
        if k["bf16"]:
            d["bf16"] += 1
        else:
            d["float_only"].append(k["name"])

    total = len(ks)
    n_bf16 = sum(1 for k in ks if k["bf16"])
    rep = {"kernels_total": total, "kernels_bf16": n_bf16,
           "kernels_float_only": total - n_bf16, "by_category": {}}
    for cat, d in by_cat.items():
        rep["by_category"][cat] = {"total": d["total"], "bf16": d["bf16"],
                                   "float_only": d["float_only"]}

    if a.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0

    print("== 真 AMP 差距清单 ==")
    print("内核总数 %d，其中支持 bf16 的 %d 个，只吃 float 的 **%d 个**"
          % (total, n_bf16, total - n_bf16))
    print("")
    print("| 类别 | 内核数 | 支持 bf16 | 只吃 float（需要加变体） |")
    print("| --- | --- | --- | --- |")
    for cat in [c[0] for c in CATEGORIES if c[0] in by_cat]:
        d = by_cat[cat]
        ex = ", ".join(d["float_only"][:4]) + ("…" if len(d["float_only"]) > 4 else "")
        print("| %s | %d | %d | %s |" % (cat, d["total"], d["bf16"], ex or "—"))
    print("")
    print("结论：matmul 走 cuBLAS 已支持 bf16（TG_BF16=1）；缺口在逐元素/归一化/注意力/CE。")
    print("第一步只需要「张量进出」的那批（逐元素 + 归一化 + 嵌入），约 %d 个。"
          % (by_cat.get("逐元素（加减乘/激活）", {}).get("total", 0)
             + by_cat.get("归一化", {}).get("total", 0)
             + by_cat.get("嵌入 gather/scatter", {}).get("total", 0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
