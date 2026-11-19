# -*- coding: utf-8 -*-
"""按 kernel 名字统计发射次数（前向 / 反向分开），定位"遍数"到底堆在哪。

不给每个 kernel 计时（5,212 次/步，逐次同步会失真），只数次数：
次数 × 单次发射固定开销(≈10 µs) 就是派发成本；次数大且张量小的 kernel，
就是"多趟访存"的嫌疑对象。
"""
import atexit
import json
import sys
from collections import Counter

import tgai_go.train as T
from tgai_go.train import v3_pretrain as TR

rt = T._cuda()
phase = {"cur": "fwd"}
cnt = {"fwd": Counter(), "bwd": Counter()}

_launch = rt.launch


def launch_counted(name, *a, **kw):
    cnt[phase["cur"]][str(name)] += 1
    return _launch(name, *a, **kw)


rt.launch = launch_counted

# 用 backward 的进出切换 phase
_T = None
for _nm in ("Tensor", "_Tensor"):
    if hasattr(T, _nm):
        _T = getattr(T, _nm)
        break
if _T is not None:
    _bwd = _T.backward

    def bwd_tagged(self, *a, **kw):
        prev = phase["cur"]
        phase["cur"] = "bwd"
        try:
            return _bwd(self, *a, **kw)
        finally:
            phase["cur"] = prev

    _T.backward = bwd_tagged


def _report():
    out = {}
    for ph in ("fwd", "bwd"):
        out[ph + "_total"] = sum(cnt[ph].values())
        out[ph + "_top"] = cnt[ph].most_common(15)
    print("[kernels] " + json.dumps(out, ensure_ascii=False), flush=True)


atexit.register(_report)

if __name__ == "__main__":
    TR.main(sys.argv[1:])
