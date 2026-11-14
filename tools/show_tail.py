# -*- coding: utf-8 -*-
"""打印文件最后 N 行（远程跑测试时用来取摘要，避免输出被截断在头部）。"""
import io
import sys


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    lines = io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    for line in lines[-n:]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
