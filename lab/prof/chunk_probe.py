# -*- coding: utf-8 -*-
"""给真实训练器打桩：记录每步的 CSA 分块统计 + 该步耗时，然后看相关性。

不猜任何内部 API —— 直接 monkeypatch `v3_csa.make_plan`，走 v3_pretrain 的真实路径。

用法（V100）：
    D:\\orv_pip_0962\\Scripts\\python.exe chunk_probe.py --layers 24 --d_ff 3328 --steps 12
"""
import argparse
import sys

import numpy as np

from tgai_go.train import v3_csa as C
from tgai_go.train import v3_pretrain as P

REC = []
_orig = C.make_plan


def make_plan(csac, B, T, seg=None, **kw):
    plan = _orig(csac, B, T, seg=seg, **kw)
    try:
        ncm = None
        try:
            ncm = np.asarray(plan._n["n_chunk"])
        except Exception:
            ncm = None
        Tc = int(getattr(plan, "Tc", -1))
        fb = bool(getattr(plan, "fallback", False))
        sg = bool(getattr(plan, "single", False))
        us = bool(getattr(plan, "use_seg", False))
        nmax = int(ncm.max()) if ncm is not None and ncm.size else -1
        tot = int(ncm.sum()) if ncm is not None and ncm.size else -1
        REC.append((nmax, tot, Tc, (int(fb), int(sg), int(us))))
    except Exception as e:
        REC.append((-1, -1, -1, str(e)[:40]))
    return plan


C.make_plan = make_plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--d_ff", type=int, default=3328)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seq", type=int, default=512)
    a = ap.parse_args()

    sys.argv = ["v3_pretrain",
                "--data", r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512",
                "--vocab", "96000", "--seq", str(a.seq), "--d", "1024",
                "--layers", str(a.layers), "--E", "4", "--n_q", "16",
                "--d_ff", str(a.d_ff), "--top_k", "2", "--csa", "4",
                "--steps", str(a.steps), "--log-every", "1", "--no-final-save"]
    try:
        P.main()
    except SystemExit:
        pass

    print("\n=== 每步 CSA 分块 vs 步时 ===")
    print(" step | n_chunk.max | 总chunk | Tc | (fallback,single,use_seg)")
    for i, r in enumerate(REC):
        print(" %4d | %11s | %7s | %s | %s" % (i, r[0], r[1], r[2], r[3]))
    arr = np.asarray([r[:2] for r in REC], dtype=np.float64)
    if len(arr) > 2:
        for j, nm in enumerate(("n_chunk.max", "总chunk")):
            x = arr[:, j]
            print("  %s: min=%d max=%d std=%.2f" % (nm, x.min(), x.max(), x.std()))


if __name__ == "__main__":
    main()
