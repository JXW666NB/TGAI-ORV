# -*- coding: utf-8 -*-
"""薄封装：V3 训练脚本的**唯一实现已移进包里** —— `tgai_go.train.v3_train_cli`。

## 为什么改

远端 V100 原先只能通过中继 `python tgai_ops.py v100py '<code>'` 执行代码，而实测单条命令
只能可靠承载约 1 KB 的 base64（600 B 成功；5.7 KB 起失败）—— 几 KB 的脚本传不过去。
做成包内模块后随 `pip install` 到达远端，直接（**推荐用这一行**）：

    python -m tgai_go.train.v3_train_cli --d 1024 --layers 20 --E 4 --seq 256 \\
        --steps 2 --top_k 2 --d_ff 2816 --vocab 96000

即使后来有了 scp 直传（2026-11-16 验证），进包仍然省掉"手工传文件 + 脚本/引擎版本错配"。

## 本文件只保留两件事（**不要在这里加训练逻辑**，否则又会两份漂移）

1. 把仓库 `python/` 放进 sys.path —— 使未 `pip install` 时
   `python tools/v3_orv_train.py ...` 仍然能跑（兼容旧命令）；
2. re-export `build` / `params_of` / `main` —— `tools/probe_ablate.py`、
   `probe_bwd_gemm.py`、`v3_orv_train_ckpt*.py` 等 11 个探针脚本都
   `from v3_orv_train import build, params_of`，**删掉这两个名字会让它们 ImportError**。

原脚本的实现（含 build()/params_of() 里"专家权重用列表不用三维索引"、"router 必须是
独立参数"等实测踩坑注释）已**逐字搬进** `python/tgai_go/train/v3_train_cli.py`。
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _cand in (os.path.join(_REPO, "python"), _REPO):
    if os.path.isdir(os.path.join(_cand, "tgai_go")) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from tgai_go.train.v3_train_cli import (  # noqa: E402,F401
    build, build_arg_parser, main, params_of)

if __name__ == "__main__":
    raise SystemExit(main())
