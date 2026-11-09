# -*- coding: utf-8 -*-
"""只打印检查点里第 0 层的键名（用于确定 load_all 的取数方式），不做任何计算。"""
import sys

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"


def main():
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState

    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    log("TGState 的公开方法: %s" % [n for n in dir(st) if not n.startswith("_")])
    names = None
    for attr in ("names", "keys", "index", "keys_list"):
        if hasattr(st, attr):
            v = getattr(st, attr)
            names = list(v() if callable(v) else v)
            log("用 st.%s 取到 %d 个名字" % (attr, len(names)))
            break
    if names is None:
        try:
            names = list(st.keys())
        except Exception as e:
            log("取名字失败: %s" % type(e).__name__)
            return
    hit = [n for n in names if ("layers.0" in n or "layers_0" in n or n.startswith("0."))][:24]
    log("第 0 层相关键名（前 24 个）:")
    for n in hit:
        log("   %s" % n)
    st.close()


if __name__ == "__main__":
    main()
