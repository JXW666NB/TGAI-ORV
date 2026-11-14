# -*- coding: utf-8 -*-
"""比较两份 pytest 输出里的 FAILED 行，找出新增/消失的失败。

用法：python tools/diff_failed.py <基线.txt> <当前.txt>
"""
import io
import sys


def failed(path):
    out = []
    for line in io.open(path, encoding="utf-8", errors="replace"):
        if line.startswith("FAILED"):
            out.append(line.strip())
    return out


def main():
    a, b = failed(sys.argv[1]), failed(sys.argv[2])
    print("基线失败 %d，当前失败 %d" % (len(a), len(b)))
    new = [x for x in b if x not in a]
    gone = [x for x in a if x not in b]
    print("\n新增（%d）：" % len(new))
    for x in new:
        print("  " + x)
    print("\n消失（%d）：" % len(gone))
    for x in gone:
        print("  " + x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
