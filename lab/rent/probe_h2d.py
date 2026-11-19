# -*- coding: utf-8 -*-
"""定位最大的 h2d 上传来自哪里（记录 nbytes + 调用栈帧）。

背景：实测每步约 299 MB 主机→设备上传，d2h 只有 2.5 MB（一侧流量），
最可能是 AMP 的 fp16 权重影子重铸。这里抓到具体调用点。
"""
import atexit
import json
import sys
import traceback
from collections import Counter

import tgai_go.train as T
from tgai_go.train import v3_pretrain as TR

rt = T._cuda()
big = Counter()          # (file:line) -> bytes
_printed = []
_h2d = rt.h2d


def h2d_probe(arr, *a, **kw):
    try:
        nb = int(getattr(arr, "nbytes", 0))
    except Exception:
        nb = 0
    if nb >= (1 << 20):                       # 只记 ≥1 MB 的调用
        st = traceback.extract_stack(limit=10)
        # 取第一个不属于本文件的帧
        site = "?"
        for fr in reversed(st):
            if "probe_h2d" not in fr.filename and "traceback" not in fr.filename:
                site = "%s:%d" % (fr.filename.split("\\")[-1], fr.lineno)
                break
        big[site] += nb
        if not _printed and "init__.py" in site:
            _printed.append(site)
            print("[h2d-stack] " + " <- ".join(
                "%s:%d" % (fr.filename.split("\\")[-1], fr.lineno) for fr in st), flush=True)
    return _h2d(arr, *a, **kw)


rt.h2d = h2d_probe


def _report():
    tot = sum(big.values())
    top = [[k, v] for k, v in big.most_common(8)]
    print("[h2d] " + json.dumps({"total_mb": tot / 1e6, "top": top},
                                ensure_ascii=False), flush=True)


atexit.register(_report)

if __name__ == "__main__":
    TR.main(sys.argv[1:])
