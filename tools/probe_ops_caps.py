# -*- coding: utf-8 -*-
"""探测设备侧块化所需的算子是否存在。"""
import sys

sys.path.insert(0, "D:/orv_pip_0962")

import tgai_go.train as T  # noqa: E402


def main():
    tn = T.Tensor
    for name in ("max", "log", "exp", "sum", "reshape", "transpose"):
        print("Tensor.%-10s %s" % (name, hasattr(tn, name)))
    for name in ("maximum", "max", "log", "cat", "exp"):
        print("module.%-10s %s" % (name, hasattr(T, name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
