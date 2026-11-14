# -*- coding: utf-8 -*-
"""打印文件中匹配子串的最后 N 行（远程取测试摘要用，避免整份输出被截断）。"""
import io
import sys


def main():
    path = sys.argv[1]
    pat = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    lines = io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    hit = [ln for ln in lines if pat in ln]
    for line in hit[-n:]:
        print(line[:200])
    if not hit:
        print("NO MATCH for %r" % pat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
