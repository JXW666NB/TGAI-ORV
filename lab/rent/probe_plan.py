# -*- coding: utf-8 -*-
"""为什么 CSA 掩码全 -1e30？打印 plan / layout 内部量。"""
import os
import sys

WORK = r"D:\orv_bench"
os.chdir(WORK)
sys.path.insert(0, WORK)

import numpy as np                                              # noqa: E402

import tgai_go.train as T                                       # noqa: E402
from tgai_go.train import v3_pretrain as TR                     # noqa: E402
from tgai_go.train import v3_csa as C                           # noqa: E402

_orig_make_plan = C.make_plan
DUMP = [False]


def _make_plan(cfg, B, Tlen, seg=None, cache=True):
    plan = _orig_make_plan(cfg, B, Tlen, seg=seg, cache=cache)
    if not DUMP[0]:
        DUMP[0] = True
        print("[plan] cfg: csa_seg_isolate=%r local_win=%r csa_compress=%r conv_k=%r"
              % (getattr(cfg, "csa_seg_isolate", None), getattr(cfg, "local_win", None),
                 getattr(cfg, "csa_compress", None), getattr(cfg, "csa_conv_kernel", None)),
              flush=True)
        print("[plan] B=%d Tlen=%d Tc=%d c=%d lk=%d use_seg=%r single=%r fallback=%r"
              % (plan.B, plan.Tlen, plan.Tc, plan.c, plan.lk, plan.use_seg,
                 plan.single, plan.fallback), flush=True)
        if seg is not None:
            s = np.asarray(seg)
            print("[plan] seg: shape=%r min=%r max=%r nuniq=%d"
                  % (s.shape, s.min(), s.max(), len(np.unique(s))), flush=True)
        lay = plan._n.get("pool", {}).get("layout")
        if lay is None:
            print("[plan] layout = None", flush=True)
        else:
            for k, v in lay.items():
                a = np.asarray(v)
                print("[plan] layout[%-9s] shape=%r min=%r max=%r"
                      % (k, a.shape, a.min() if a.size else None,
                         a.max() if a.size else None), flush=True)
            cnt = np.asarray(lay["cnt"]).reshape(-1)
            sd = np.asarray(lay["slot_doc"]).reshape(-1)
            print("[plan]   cnt>0 的槽位 = %d / %d ; slot_doc>=0 的槽位 = %d"
                  % (int((cnt > 0).sum()), cnt.size, int((sd >= 0).sum())), flush=True)
            cid = np.asarray(lay["cid"])
            print("[plan]   cid[0,:8]=%r cid[0,-8:]=%r"
                  % (cid[0, :8].tolist(), cid[0, -8:].tolist()), flush=True)
        for key in ("mask_g", "row_g", "mask_l", "row_l"):
            a = plan._n.get(key)
            if a is None:
                print("[plan] %s = None" % key, flush=True)
                continue
            a = np.asarray(a).reshape(-1)
            print("[plan] %s: n=%d frac(==0)=%.4f frac(<-1e29)=%.4f"
                  % (key, a.size, float((a == 0).mean()), float((a < -1e29).mean())),
                  flush=True)
    return plan


C.make_plan = _make_plan

argv = ["--data", os.path.join(WORK, "data2048_v96000"), "--vocab", "96000",
        "--seq", "2048", "--d", "1024", "--layers", "12", "--E", "4",
        "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
        "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
        "--steps", "2", "--log-every", "99", "--no-final-save"]
TR.main(argv)
