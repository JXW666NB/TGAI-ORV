# -*- coding: utf-8 -*-
"""`v3_pretrain` 的 `--tie` 注入壳（**只为测量**，不改包内代码/语义）。

为什么要有它：`python -m tgai_go.train.v3_pretrain` 的 argparse 里**没有** `--tie`
（那是 `v3_train_cli` 的开关，见 `v3_train_cli.py:154` 的 `ARCH_ITEMS`），
而 `_expand_cfg()` 也不写 `cfg["tie"]`。所以基线跑的一直是**不 tie**。

本壳只做一件事：拿到 `argparse.Namespace` 后，在 `_expand_cfg` 的产物上补一个
`cfg["tie"]=True`，然后调用**原封不动的** `v3_pretrain.main()` —— 训练循环、前向、
损失、优化器全部复用包内实现，一个字节都没改。

开关：环境变量 `ORV_TIE=1` 打开（其它值/缺省 = 关闭 = 与 `python -m` 完全一致）。

用法（远端）：
    set ORV_TIE=1
    python tie_main.py --data D:\\orv_bench\\data2048_v96000 --vocab 96000 --seq 512 ...
"""
import os
import sys

from tgai_go.train import v3_pretrain as VP

_TIE = os.environ.get("ORV_TIE", "") not in ("", "0", "false", "False")
_orig_expand = VP._expand_cfg


def _expand_cfg(a):
    cfg = _orig_expand(a)
    if _TIE:
        # 与 v3_train_cli.py:349-353 完全同义：P["head"] is P["emb"]
        cfg["tie"] = True
    return cfg


VP._expand_cfg = _expand_cfg

if __name__ == "__main__":
    print("[tie_main] ORV_TIE=%r" % (_TIE,), flush=True)
    raise SystemExit(VP.main(sys.argv[1:]))
