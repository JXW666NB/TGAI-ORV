# -*- coding: utf-8 -*-
"""阶段 2 远端总调度：先跑「数据 vs 词表」判别，再跑 event 归因矩阵。

两个子脚本各自独立可跑；这里只是为了**串行**占用 GPU（同一时刻只能有一个基准）。
"""
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import ab_vocab_data          # noqa: E402
import run_attr_all          # noqa: E402


def main():
    print("########## 阶段 2A：data vs vocab 判别 ##########", flush=True)
    ab_vocab_data.main()
    print("########## 阶段 2B：event 归因矩阵 ##########", flush=True)
    run_attr_all.main()
    print("########## 阶段 2 完成 ##########", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
