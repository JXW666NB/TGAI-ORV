# -*- coding: utf-8 -*-
"""打印远端实际安装的 v3_ops 里 sit_glu_numpy 的源码，确认是否与本地一致。"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tgai_go.train import v3_ops as V  # noqa: E402


def main():
    print("模块文件:", V.__file__)
    print("有 situ_glu_numpy:", hasattr(V, "situ_glu_numpy"))
    if hasattr(V, "situ_glu_numpy"):
        print(inspect.getsource(V.situ_glu_numpy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
