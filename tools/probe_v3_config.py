# -*- coding: utf-8 -*-
"""打印真实 V3 训练脚本里的架构配置行（只读）。"""
import io
import re
import sys

PAT = re.compile(r"TGAIV3Config|n_activated|n_shared|n_heads|n_kv|d_model|n_layers|"
                 r"csa|block_attn|attn_res|top_k|vocab_size|seq_len", re.I)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "D:/TGAI-V3-1.5B/03_src/train_v3.py"
    try:
        src = io.open(path, encoding="utf-8", errors="replace").read()
    except OSError as exc:
        print("读不到 %s: %s" % (path, exc))
        return 1
    n = 0
    for i, line in enumerate(src.splitlines(), 1):
        if PAT.search(line) and not line.strip().startswith("#"):
            print("%5d: %s" % (i, line.strip()[:150]))
            n += 1
            if n >= 40:
                break
    print("—— 共打印 %d 行（上限 40）" % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
