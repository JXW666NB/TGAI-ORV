# -*- coding: utf-8 -*-
"""抓 fp16 GEMM 的完整报错（不吞异常）。"""
import os
import sys
import traceback

os.environ["TG_FP16"] = "1"
sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log


def main():
    import numpy as np
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    a = T.tensor(np.random.randn(512, 1024).astype(np.float32))
    b = T.tensor(np.random.randn(1024, 4096).astype(np.float32))
    log("fp16 状态: %r" % (rt._fp16_state(),))
    try:
        y = a @ b
        rt.sync()
        log("OK shape=%s" % (tuple(y.shape),))
    except Exception:
        log("完整堆栈：")
        for line in traceback.format_exc().splitlines()[-14:]:
            log("   " + line)


if __name__ == "__main__":
    main()
