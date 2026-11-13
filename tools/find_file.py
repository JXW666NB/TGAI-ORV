# -*- coding: utf-8 -*-
"""按文件名子串在若干根目录里找文件（远端排查用，避免在 shell 里拼 findstr）。

用法：`python find_file.py <子串> [根...]`
默认根：D:/tgai_nlp, D:/tgai_stage2, D:/tgai, D:/模型库
"""
import os
import sys

DEFAULT_ROOTS = ["D:/tgai_nlp", "D:/tgai_stage2", "D:/tgai", "D:/模型库"]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    needle = sys.argv[1].lower()
    roots = sys.argv[2:] or DEFAULT_ROOTS
    hits = 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # 跳过明显无关的大目录，控制耗时
            dirnames[:] = [d for d in dirnames
                           if d.lower() not in ("node_modules", ".git", "__pycache__")]
            for fn in filenames:
                if needle in fn.lower():
                    p = os.path.join(dirpath, fn)
                    try:
                        sz = os.path.getsize(p)
                    except OSError:
                        sz = -1
                    print("%10.1f MB  %s" % (sz / 1e6, p))
                    hits += 1
                    if hits >= 40:
                        print("...（已截断）")
                        return 0
    if not hits:
        print("未找到包含 %r 的文件" % needle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
