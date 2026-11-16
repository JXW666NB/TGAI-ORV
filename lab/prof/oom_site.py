# -*- coding: utf-8 -*-
"""定位 seq1024 OOM 到底发生在哪一段（build / plan / 层循环 / head）。

做法：给 `_raise_alloc_failed` 挂桩，失败时打印 Python 调用栈，然后照常抛出。
不改库，纯驱动脚本。
"""
import argparse
import sys
import traceback

import tgai_go.train as T
from tgai_go.train import v3_pretrain as P

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)

_orig = CLS._raise_alloc_failed


def hook(self, bucket, rc):
    print("\n===== OOM 现场调用栈（请求 %d 字节）=====" % int(bucket), flush=True)
    traceback.print_stack()
    print("===== 调用栈结束 =====", flush=True)
    return _orig(self, bucket, rc)


CLS._raise_alloc_failed = hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--d_ff", type=int, default=3328)
    ap.add_argument("--ckpt", action="store_true")
    a = ap.parse_args()

    argv = ["v3_pretrain",
            "--data", r"D:\TGAI-V3-1.5B\01_corpus\packed_1024" if a.seq == 1024
            else r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512",
            "--vocab", "96000", "--seq", str(a.seq), "--d", "1024",
            "--layers", str(a.layers), "--E", "4", "--n_q", "16",
            "--d_ff", str(a.d_ff), "--top_k", "2", "--csa", "4",
            "--steps", "1", "--log-every", "1", "--no-final-save"]
    if a.ckpt:
        argv.append("--grad-ckpt")
    sys.argv = argv
    try:
        P.main()
    except SystemExit:
        pass
    except Exception as e:
        print("\n[exc] %r" % (e,))


if __name__ == "__main__":
    main()
