# -*- coding: utf-8 -*-
"""打印某文件里一个类的源码（只读）。用法：probe_dump_class.py <file> <Class> [行数]"""
import io
import sys


def main():
    path = sys.argv[1]
    name = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 90
    lines = io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    start = [i for i, l in enumerate(lines) if l.lstrip().startswith("class " + name)]
    if not start:
        print("没找到 class %s" % name)
        return 1
    i = start[0]
    for k in range(i, min(i + n, len(lines))):
        print("%4d %s" % (k + 1, lines[k].rstrip()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
