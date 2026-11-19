# -*- coding: utf-8 -*-
"""vocab 成本归因：CUDA event 打点（**不用 sync 包裹**），按逻辑算子分段。

口径与 `bench_stable.py` 完全一致，只把步数降到 12、跳过前 4 步（`ATTR_SKIP` 可调）：
  12 层 / d=1024 / d_ff=3328 / E=4(+1 共享) / top_k=2 / n_q=16 / csa=4 /
  batch=1 / --opt int8gpu --amp gemm / ckpt 关 / seq 与 vocab 由命令行给。

打点方式（关键：每个逻辑项一对 event，成对读 elapsed；**没有一处 sync 包裹**）
  * `TR.forward_loss`          整体前向                    → fwd
  * `Tensor._matmul_cuda`      命中 head（lazy-T 的 N==vocab）→ head_fwd（并包住
                               返回张量的 `_backward` → head_bwd）
  * `Tensor._array_index_cuda` 命中 emb（形状 (vocab,d)）    → emb_fwd + emb_bwd
  * `_cross_entropy_cuda`      损失                          → ce_fwd + ce_bwd
  * `T.clip_grad_norm_`                                      → clip
  * `Int8AdamWGpu.step`                                      → opt
  * `Tensor.backward`          整体反向                      → bwd
  * `rt.launch(name=cast_f32_to_f16)` → cast（**默认只计数**；
                               `ATTR_CAST=1` 才逐次打点，因为它是 900+ 次/步，
                               逐次 event 会往主机侧加重 ~10 ms）
  * `rt.sync()`：训练器每步末/反向末会调，之后**批量**读 elapsed ⇒ 不阻塞发射流水

不改训练语义：所有 patch 都是「原函数前后各记一个 event + 原样转发返回值和异常」。
`loss`/梯度/参数与不打点时逐位相同（同一 seed 下 loss 序列可与基线对照）。

产出：stdout 的 `[attr] {...}`，以及 `ATTR_OUT` 指向的 json。
"""
import atexit
import json
import os
import sys
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_pretrain as TR

RT = T._cuda()
SKIP = int(os.environ.get("ATTR_SKIP", "4"))
OUT = os.environ.get("ATTR_OUT", r"D:\orv_bench\attr_out.json")
TAG = os.environ.get("ATTR_TAG", "")
DO_CAST = os.environ.get("ATTR_CAST", "0") not in ("0", "false", "False")
TIE = os.environ.get("ORV_TIE", "") not in ("", "0", "false", "False")

_FREE = []
_PEND = []            # [(sec, ev0, ev1)]
_SUM = {}             # sec -> [n, gpu_ms]
_CPU = {}             # sec -> [n, cpu_ms]
STEP = [-1]
NMEAS = [0]
WALL = []
_TLAST = [None]
KER = {}
VO = {"vocab": None, "seq": None}
_FAST = {}


def _init_fast():
    _FAST["rec"] = RT.cu.cuEventRecord
    _FAST["stream"] = RT._stream


def _take():
    if _FREE:
        return _FREE.pop()
    ev = RT.event_create()
    return ev


def _mark():
    if not _FAST:
        _init_fast()
    ev = _take()
    if _FAST["rec"](ev, _FAST["stream"]) != 0:
        raise RuntimeError("cuEventRecord failed")
    return ev


def _give(ev):
    _FREE.append(ev)


def _drain():
    if not _PEND:
        return
    vals, ok = [], True
    for _sec, e0, e1 in _PEND:
        try:
            vals.append(RT.event_elapsed_ms(e0, e1))
        except Exception:
            ok = False
            break
    if ok:
        for i, (sec, _e0, _e1) in enumerate(_PEND):
            st = _SUM.get(sec)
            if st is None:
                _SUM[sec] = st = [0, 0.0]
            st[0] += 1
            st[1] += vals[i]
    for _sec, e0, e1 in _PEND:
        _give(e0)
        _give(e1)
    del _PEND[:]


class _Sec(object):
    __slots__ = ("name", "e0", "t0")

    def __init__(self, name):
        self.name = name
        self.e0 = None
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.e0 = _mark()
        return self

    def __exit__(self, *a):
        e1 = _mark()
        st = _CPU.get(self.name)
        if st is None:
            _CPU[self.name] = st = [0, 0.0]
        st[0] += 1
        st[1] += (time.perf_counter() - self.t0) * 1000.0
        _PEND.append((self.name, self.e0, e1))
        return False


def _wrap_bwd(t, sec):
    orig = getattr(t, "_backward", None)
    if orig is None:
        return

    def wb(g, grads, _o=orig):
        with _Sec(sec):
            _o(g, grads)

    t._backward = wb


def _on():
    return STEP[0] >= SKIP


# ── 1) 整体前向 / 反向 ────────────────────────────────────────────────────
_orig_fwd = TR.forward_loss
_orig_bwd = T.Tensor.backward


def _fwd(*a, **kw):
    STEP[0] += 1
    now = time.perf_counter()
    if _TLAST[0] is not None:
        WALL.append(now - _TLAST[0])
    _TLAST[0] = now
    if not _on():
        return _orig_fwd(*a, **kw)
    NMEAS[0] += 1
    with _Sec("fwd"):
        return _orig_fwd(*a, **kw)


def _bwd(self, *a, **kw):
    if not _on():
        return _orig_bwd(self, *a, **kw)
    with _Sec("bwd"):
        return _orig_bwd(self, *a, **kw)


TR.forward_loss = _fwd
T.Tensor.backward = _bwd

# ── 2) head GEMM ─────────────────────────────────────────────────────────
_orig_mm = T.Tensor._matmul_cuda


def _mm(self, other):
    src = getattr(other, "_lazy_T", None)
    is_head = (src is not None and VO["vocab"] is not None
               and int(src.shape[0]) == int(VO["vocab"]) and int(self.ndim) == 2)
    if not (is_head and _on()):
        return _orig_mm(self, other)
    with _Sec("head_fwd"):
        out = _orig_mm(self, other)
    _wrap_bwd(out, "head_bwd")
    return out


T.Tensor._matmul_cuda = _mm

# ── 3) embedding gather ──────────────────────────────────────────────────
_orig_idx = T.Tensor._array_index_cuda


def _idx(self, arr):
    is_emb = (VO["vocab"] is not None and int(self.ndim) == 2
              and int(self.shape[0]) == int(VO["vocab"]))
    if not (is_emb and _on()):
        return _orig_idx(self, arr)
    with _Sec("emb_fwd"):
        out = _orig_idx(self, arr)
    _wrap_bwd(out, "emb_bwd")
    return out


T.Tensor._array_index_cuda = _idx

# ── 4) cross-entropy ─────────────────────────────────────────────────────
_orig_ce = T._cross_entropy_cuda


def _ce(logits, target, ignore_index=-100):
    if not _on():
        return _orig_ce(logits, target, ignore_index)
    with _Sec("ce_fwd"):
        loss = _orig_ce(logits, target, ignore_index)
    _wrap_bwd(loss, "ce_bwd")
    return loss


T._cross_entropy_cuda = _ce

# ── 5) clip / optimizer ──────────────────────────────────────────────────
_orig_clip = T.clip_grad_norm_


def _clip(*a, **kw):
    if not _on():
        return _orig_clip(*a, **kw)
    with _Sec("clip"):
        return _orig_clip(*a, **kw)


T.clip_grad_norm_ = _clip

try:
    from tgai_go.train.int8_adamw_gpu import Int8AdamWGpu

    _orig_opt = Int8AdamWGpu.step

    def _opt(self):
        if not _on():
            return _orig_opt(self)
        with _Sec("opt"):
            return _orig_opt(self)

    Int8AdamWGpu.step = _opt
except Exception as _e:                                    # noqa: BLE001
    print("[attr] no int8gpu hook: %r" % (_e,), flush=True)

# ── 6) cast kernel ───────────────────────────────────────────────────────
_orig_launch = RT.launch


def _launch(name, grid, block, args, shared=0, module=None):
    if name != "cast_f32_to_f16":
        return _orig_launch(name, grid, block, args, shared, module)
    if not _on():
        return _orig_launch(name, grid, block, args, shared, module)
    KER[name] = KER.get(name, 0) + 1
    if not DO_CAST:
        return _orig_launch(name, grid, block, args, shared, module)
    with _Sec("cast"):
        return _orig_launch(name, grid, block, args, shared, module)


RT.launch = _launch

# ── 7) 每步末批量读 event；抓 vocab/seq；tie 注入 ─────────────────────────
_orig_sync = RT.sync


def _sync():
    _orig_sync()
    _drain()


RT.sync = _sync

_orig_expand = TR._expand_cfg


def _expand_cfg(a):
    cfg = _orig_expand(a)
    VO["vocab"] = int(cfg["vocab"])
    VO["seq"] = int(cfg["seq"])
    if TIE:
        cfg["tie"] = True
    _init_fast()
    return cfg


TR._expand_cfg = _expand_cfg


def _report():
    _drain()
    n = max(1, NMEAS[0])
    sec = {k: {"n_per_step": v[0] / float(n), "ms_per_step": v[1] / float(n)}
           for k, v in _SUM.items()}
    cpu = {k: {"n_per_step": v[0] / float(n), "cpu_ms_per_step": v[1] / float(n)}
           for k, v in _CPU.items()}
    wall = WALL[SKIP:] if len(WALL) > SKIP else WALL
    res = {"tag": TAG, "tie": TIE, "vocab": VO["vocab"], "seq": VO["seq"],
           "measured_steps": NMEAS[0], "skipped": SKIP,
           "step_s_wall_mean": float(np.mean(wall)) if wall else None,
           "step_s_wall": [round(x, 4) for x in wall],
           "sections": sec, "cpu": cpu,
           "cast_kernels_per_step": KER.get("cast_f32_to_f16", 0) / float(n)}
    print("[attr] " + json.dumps(res, ensure_ascii=False), flush=True)
    try:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("[attr] wrote %s" % OUT, flush=True)
    except Exception as e:                                # noqa: BLE001
        print("[attr] write failed %r" % (e,), flush=True)


atexit.register(_report)

if __name__ == "__main__":
    print("[attr] tie=%r cast=%r skip=%d out=%s" % (TIE, DO_CAST, SKIP, OUT), flush=True)
    raise SystemExit(TR.main(sys.argv[1:]))
