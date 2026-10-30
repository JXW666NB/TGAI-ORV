# -*- coding: utf-8 -*-
"""后端接入体检：给一个候选后端出「核心 / 可回退 / 可选」三层报告。

用途：写 Vulkan / ROCm 后端时，先跑这个，它告诉你**还差哪些必须实现**、
哪些可以先跳过（有回退路径）、哪些是 MoE 专属。

用法：
    python tools/backend_port.py numpy_ref                    # 内置参考后端
    python tools/backend_port.py my_pkg.my_mod:make           # 你的工厂函数
    python tools/backend_port.py my_pkg.my_mod:MyBackend      # 或直接给类（无参构造）
"""
import argparse
import importlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)

from tgai_go.train import backend  # noqa: E402


def resolve(spec):
    """解析 'name' | 'module:attr' 得到后端实例。"""
    if spec == "numpy_ref":
        from tgai_go.train.backends import numpy_ref
        return numpy_ref.make()
    if ":" not in spec:
        raise SystemExit(
            "无法解析 %r。用法：内置用 'numpy_ref'，自定义用 'module:factory' 或 'module:Class'"
            % (spec,))
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    return obj() if callable(obj) else obj


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", default="numpy_ref")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    inst = resolve(a.spec)
    r = backend.report(inst)

    print("后端：%s" % r["name"])
    print("=" * 66)
    print("  核心（dense 训练必做）      %2d / %2d" % r["core"])
    print("  可回退（可先不实现）        %2d / %2d" % r["fallback"])
    print("  可选 MoE                    %2d / %2d" % r["optional"])
    print("  能训 dense 模型             %s" % ("是" if r["trainable_dense"] else "否"))
    print()

    if r["missing_core"]:
        print("还缺的核心方法（必须补，否则 dense 训练跑不起来）：")
        for i in range(0, len(r["missing_core"]), 4):
            print("   " + "  ".join("%-26s" % m for m in r["missing_core"][i:i + 4]))
    else:
        print("核心方法已齐 —— 可以尝试真正跑一次训练。")

    if a.verbose:
        if r["missing_fallback"]:
            print("\n可回退组缺失（走组合路径，会慢）：")
            print("   " + ", ".join(r["missing_fallback"]))
        if r["missing_optional"]:
            print("\nMoE 缺失（不用 MoE 可忽略）：")
            print("   " + ", ".join(r["missing_optional"]))

    print()
    print("下一步：")
    print("  1) 补齐「还缺的核心方法」（参考 python/tgai_go/train/backends/numpy_ref.py 的写法）")
    print("  2) 用 TG_BACKEND=<你的名字> python -m tgai_go.train.doctor 跑自检")
    print("  3) 数值对齐：与 cuda 后端同输入比 relerr ≤1e-5（见 docs/BACKEND_PORTING.md）")
    return 0 if r["trainable_dense"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
