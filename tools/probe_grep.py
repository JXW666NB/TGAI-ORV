# -*- coding: utf-8 -*-
"""在目录内搜架构配置（只读）。用法：probe_grep.py <dir> [pattern]"""
import io
import os
import re
import sys

DEFAULT = r"n_activated|n_shared_experts|use_csa|block_attn|attn_res|n_experts|top_k"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "D:/TGAI-V3-1.5B/03_src"
    pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else DEFAULT, re.I)
    hits = 0
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(root, name)
        try:
            src = io.open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if pat.search(line):
                print("%s:%d: %s" % (name, i, line.strip()[:140]))
                hits += 1
                if hits >= 40:
                    print("—— 达到 40 行上限")
                    return 0
    print("—— 共 %d 行" % hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
