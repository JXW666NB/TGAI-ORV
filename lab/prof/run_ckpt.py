# -*- coding: utf-8 -*-
"""确认 `TG_GRAD_CKPT` 真的让重算生效：同一驱动，开/关各跑一次。

若重算生效：step 时间应上升（每层多一次前向），且显存应下降。
用脚本设环境变量，避免 PowerShell 的引号问题。
"""
import argparse
import os
import sys

import tgai_go.train as T
from tgai_go.train import v3_pretrain as P

T.set_default_device("cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    a = ap.parse_args()

    if a.ckpt:
        os.environ["TG_GRAD_CKPT"] = "1"
    else:
        os.environ.pop("TG_GRAD_CKPT", None)

    sys.argv = ["v3_pretrain",
                "--data", r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512",
                "--vocab", "96000", "--seq", str(a.seq), "--d", "1024",
                "--layers", str(a.layers), "--E", "4", "--n_q", "16",
                "--d_ff", "2816", "--top_k", "2", "--csa", "4",
                "--steps", "4", "--log-every", "1", "--no-final-save"]
    # 内核计数：重算会让每步多出一整次前向的内核 ⇒ 计数必然上升。
    # 这比时间更可靠（时间受运行先后/频率影响）。
    CLS = type(T._cuda())
    CNT = [0]
    _ol = CLS.launch

    def launch(self, name, grid, block, args=None, *ar, **kw):
        CNT[0] += 1
        return _ol(self, name, grid, block, args, *ar, **kw)

    CLS.launch = launch

    try:
        P.main()
    except SystemExit:
        pass
    print("[kernels] total = %d  (ckpt=%d)" % (CNT[0], a.ckpt))


if __name__ == "__main__":
    main()
