# -*- coding: utf-8 -*-
"""按行号区间打印源码（只读）。用法：probe_lines.py <file> <start> <count>"""
import io
import sys


def main():
    path, start, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    lines = io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    for i in range(max(0, start - 1), min(start - 1 + count, len(lines))):
        print("%5d %s" % (i + 1, lines[i].rstrip()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
