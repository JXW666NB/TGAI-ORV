# -*- coding: utf-8 -*-
"""列出若干目录的内容（远端排查用，避免在 shell 里拼 `&` 之类被解析掉）。

用法：`python ls_dirs.py D:/a D:/b`
"""
import os
import sys


def main():
    for d in sys.argv[1:]:
        print("=== %s" % d)
        if not os.path.isdir(d):
            print("   (不是目录)")
            continue
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if os.path.isdir(p):
                print("   [dir ] %s" % n)
            else:
                print("   [file] %-40s %10d" % (n, os.path.getsize(p)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
