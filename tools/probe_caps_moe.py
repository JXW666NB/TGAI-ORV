# -*- coding: utf-8 -*-
"""确认 ORV 是否支持 MoE 合并分派所需的两个能力：3D reshape + 中间维 sum。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def main():
    T.set_default_device("cuda")
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = T.tensor(a)

    ok = {}
    try:
        r = t.sum(axis=1)
        ok["sum(axis=1)"] = float(np.abs(np.asarray(r.to_numpy()) - a.sum(1)).max())
    except Exception as exc:
        ok["sum(axis=1)"] = "ERR: %s" % str(exc)[:80]
    try:
        r = t.reshape(6, 4)
        ok["reshape(6,4)"] = float(np.abs(np.asarray(r.to_numpy()) - a.reshape(6, 4)).max())
    except Exception as exc:
        ok["reshape(6,4)"] = "ERR: %s" % str(exc)[:80]
    try:
        idx = np.array([0, 2, 4, 1], np.int64)
        r = t.reshape(6, 4)[idx]
        ok["gather rows"] = float(np.abs(np.asarray(r.to_numpy()) - a.reshape(6, 4)[idx]).max())
    except Exception as exc:
        ok["gather rows"] = "ERR: %s" % str(exc)[:80]
    for k, v in ok.items():
        print("%-14s %s" % (k, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
