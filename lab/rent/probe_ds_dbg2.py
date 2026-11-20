# -*- coding: utf-8 -*-
"""第二步定位：CSA layer0 全局分支到底哪一步变成 0。

打印：mask/row_mask 的统计、p 的零/均值、p@v（mask 之前）的 absmax、
numpy(fp64) 复算的同一个 GEMM、以及 mask 之后的 absmax。
"""
import os
import sys

WORK = r"D:\orv_bench"
os.chdir(WORK)
sys.path.insert(0, WORK)

import numpy as np                                              # noqa: E402

import tgai_go.train as T                                       # noqa: E402
from tgai_go.train import optim as _optim                       # noqa: E402
from tgai_go.train import v3_pretrain as TR                     # noqa: E402
from tgai_go.train import v3_csa as C                           # noqa: E402

STEP = [0]
CUR = {"step": 0, "csa": 0}
TARGET = {2}
_CSA_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10]
DONE = [False]


def _np(buf):
    return np.asarray(T.Tensor._from_gpu(buf, requires_grad=False).to_numpy(), np.float32)


def _summ(a, tag):
    a = np.asarray(a, np.float64).reshape(-1)
    print("      %-10s n=%d absmax=%.6e mean=%.6e std=%.6e frac0=%.4f"
          % (tag, a.size, np.abs(a).max(), a.mean(), a.std(),
             float(np.count_nonzero(a == 0)) / a.size), flush=True)


_orig_attn = C._attn


def _attn_probe(q, k, v, mask, row_mask, scale):
    idx = CUR["csa"]
    CUR["csa"] = idx + 1
    li = idx // 2
    layer = _CSA_LAYERS[li] if li < len(_CSA_LAYERS) else -1
    local = (idx % 2 == 1)
    want = (CUR["step"] in TARGET) and (layer == 0) and (not local) and (not DONE[0])
    if want:
        DONE[0] = True
        print("   [fw] csa l0 global:", flush=True)
        _summ(_np(mask._gpu), "mask")
        if row_mask is not None:
            _summ(_np(row_mask._gpu), "row_mask")
        else:
            print("      row_mask = None", flush=True)
        _summ(_np(q._gpu), "q")
        _summ(_np(k._gpu), "k")
        _summ(_np(v._gpu), "v")
    scores = (q @ k.swap_last2()) * float(scale)
    scores = scores + mask
    p = T.functional.softmax(scores, axis=-1)
    if want:
        pn = _np(p._gpu)
        _summ(pn, "p")
        _summ(_np(scores._gpu), "scores")
        out0 = p @ v
        o0 = _np(out0._gpu)
        _summ(o0, "p@v(mask前)")
        vn = _np(v._gpu)
        ref = np.einsum("bhas,bhsd->bhad", pn.astype(np.float64),
                        vn.astype(np.float64))
        print("      fp64 复算 p@v absmax=%.6e  与 GPU 差=%.3e"
              % (np.abs(ref).max(), np.abs(ref - o0).max()), flush=True)
        if row_mask is not None:
            rn = _np(row_mask._gpu)
            print("      row_mask 非零行数=%d / %d"
                  % (int(np.count_nonzero(rn.reshape(-1, rn.shape[-1])[:, 0])),
                     rn.size), flush=True)
        print("      grad 侧 out_absmax 见下", flush=True)
    out = p @ v
    if row_mask is not None:
        out = out * row_mask
    return out


C._attn = _attn_probe

_orig_sched_step = _optim.lr_scheduler.CosineLR.step


def _sched_step(self):
    STEP[0] += 1
    return _orig_sched_step(self)


_optim.lr_scheduler.CosineLR.step = _sched_step

_orig_fwd = TR.forward_loss


def _fwd(P, cfg, kinds, cos, sin, ids, labels, seg, **kw):
    CUR["step"] = STEP[0] + 1
    CUR["csa"] = 0
    loss, meta = _orig_fwd(P, cfg, kinds, cos, sin, ids, labels, seg, **kw)
    if CUR["step"] in TARGET:
        print("[fwd] step %d loss %.6f  seg=%r"
              % (CUR["step"], float(np.asarray(loss.to_numpy()).reshape(-1)[0]),
                 (None if seg is None else tuple(seg.shape))), flush=True)
    return loss, meta


TR.forward_loss = _fwd

argv = ["--data", os.path.join(WORK, "data2048_v96000"), "--vocab", "96000",
        "--seq", "2048", "--d", "1024", "--layers", "12", "--E", "4",
        "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
        "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
        "--steps", "3", "--log-every", "99", "--no-final-save"]
TR.main(argv)
