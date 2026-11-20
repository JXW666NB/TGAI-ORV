# -*- coding: utf-8 -*-
"""从**真实训练**里 dump 注意力反向的 dS = dL/dscores 分布（不改源码，全 monkeypatch）。

背景（上一轮实测）：dS 的 absmax ~2.6e-5，远低于 fp16 最小正规数 6.1e-5；
朴素 cast 后 dq/dk 的 relFro = 0.38。要判断「先乘 2^k 再 cast」是否可行，
必须先知道 dS **随训练进展**怎么变、以及各层差多少。

本脚本（全部 monkeypatch，不改 `python/tgai_go/`）：
  1. 包 `tgai_go.train._attn_scores`（3 个 global 层的物化 SDPA 打分）：
     在它的 backward 里拿到**真正喂给 dq/dk GEMM 的那个 fp32 缓冲**。
  2. 包 `tgai_go.train._softmax_cuda`：给 global 层的 dp = dL/dp 插捕获节点。
  3. 包 `v3_csa._attn` 成同构副本，在 `scores+mask` 之后 / `softmax` 之后各插一个
     恒等捕获节点（clone + 自定义 _backward），分别拿 dS / dp。
     默认 `_ATTN_CQ=[2048]`、Lq=2048 ⇒ `_attn_chunked` 直接回落 `_attn`，路径一致。
  4. **修一个上游 mask bug**（可选，cfg.fix_csa_mask）：`_chunk_layout` 的 `slot_doc`
     存的是**文档序号（1 起）**，而 `_mask_global_seg` 拿它跟**原始 seg 值**比
     （`cseg[k] == seg[i]`）。当 seg 全 0（`--data` 的 npz 里没有 seg_id 时就是全 0）
     ⇒ 0 != 1 ⇒ 整张掩码全 -1e30、row_mask 全 0 ⇒ **CSA 注意力完全死掉**（实测）。
     这里改成用 `layout["doc_ord"]` 比，恢复参考实现的语义。
  5. 用 `CosineLR.step` 切步（每步恰好一次），只在目标 step 捕获。

产物：D:\\orv_bench\\ds_dist.json + ds_samples/*.npy
跑法（远端）：D:\\orv_bench\\remote_py.bat D:\\orv_bench\\probe_ds_dist.py
"""
import atexit
import json
import os
import sys
import traceback

WORK = r"D:\orv_bench"
OUT = os.path.join(WORK, "ds_dist.json")
SAMP = os.path.join(WORK, "ds_samples")
os.makedirs(SAMP, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

import numpy as np                                              # noqa: E402

import tgai_go.train as T                                       # noqa: E402
from tgai_go.train import optim as _optim                       # noqa: E402
from tgai_go.train import v3_pretrain as TR                     # noqa: E402
from tgai_go.train import v3_csa as C                           # noqa: E402

RT = T._cuda()

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05
FP16_MIN_SUB = 5.960464477539063e-08

_CFG = {}
_cp = os.path.join(WORK, "ds_cfg.json")
if os.path.exists(_cp):
    with open(_cp, encoding="utf-8-sig") as _f:
        _CFG = json.load(_f)
STEPS = int(_CFG.get("steps", 200))
TARGET = set(int(x) for x in str(_CFG.get("target", "2,10,50,200")).split(","))
FIX_MASK = bool(_CFG.get("fix_csa_mask", True))
print("[probe] cfg = %r" % (_CFG,), flush=True)

STEP = [0]
CUR = {"step": 0, "sdpa": 0, "csa": 0, "sm": 0}
ROWS = []
LOSS = {}
_RNG = np.random.default_rng(2026)
_SUB = 2_000_000
NEG_BIG = -1.0e30


# ---------------------------------------------------------------- 掩码修正
def _mask_global_seg_fixed(seg, layout, Tc):
    seg = np.asarray(seg, np.int32)
    B, Tlen = seg.shape
    cnt = layout["cnt"]
    cseg = layout["slot_doc"].reshape(B, Tc)
    valid_chunk = (cnt > 0).reshape(B, Tc) & (cseg >= 0)
    do = np.asarray(layout["doc_ord"])                     # (B,T) 1 起序号
    cid = layout["cid"]
    ar = np.arange(Tc)[None, None, :]
    cmask = ((cseg.reshape(B, 1, Tc) == do.reshape(B, Tlen, 1))
             & valid_chunk.reshape(B, 1, Tc)
             & (ar <= cid.reshape(B, Tlen, 1)))
    m = np.where(cmask, 0.0, NEG_BIG).astype(np.float32)
    row = cmask.any(axis=-1).astype(np.float32)
    return (np.ascontiguousarray(m.reshape(B, 1, Tlen, Tc)),
            np.ascontiguousarray(row.reshape(B, 1, Tlen, 1)))


def _mask_local_seg_fixed(seg, layout, Tc, lk):
    seg = np.asarray(seg, np.int32)
    B, Tlen = seg.shape
    cnt = layout["cnt"]
    cseg = layout["slot_doc"].reshape(B, Tc)
    valid_chunk = (cnt > 0).reshape(B, Tc) & (cseg >= 0)
    do = np.asarray(layout["doc_ord"])
    cid = layout["cid"]
    cl = cseg[:, Tc - lk:]
    cl_ok = valid_chunk[:, Tc - lk:]
    key_chunk = np.arange(Tc - lk, Tc)[None, None, :]
    iq = np.arange(Tlen)[None, :, None]
    right_aligned = np.arange(lk)[None, None, :] < (iq + (lk - Tlen + 1))
    same = ((do.reshape(B, Tlen, 1) == cl.reshape(B, 1, lk))
            & (seg >= 0).reshape(B, Tlen, 1)
            & cl_ok.reshape(B, 1, lk))
    vis = same & (key_chunk <= cid.reshape(B, Tlen, 1)) & right_aligned
    m = np.where(vis, 0.0, NEG_BIG).astype(np.float32)
    row = vis.any(axis=-1).astype(np.float32)
    return (np.ascontiguousarray(m.reshape(B, 1, Tlen, lk)),
            np.ascontiguousarray(row.reshape(B, 1, Tlen, 1)))


if FIX_MASK:
    C._mask_global_seg = _mask_global_seg_fixed
    C._mask_local_seg = _mask_local_seg_fixed
    print("[probe] CSA mask 已修正（slot_doc 序号 vs seg 原值）", flush=True)


# ---------------------------------------------------------------- 统计 / 捕获
def _stats(a, tag, extra=None):
    flat = a.reshape(-1)
    n = int(flat.size)
    aa = np.abs(flat)
    absmax = float(aa.max()) if n else 0.0
    s = aa[_RNG.integers(0, n, _SUB)] if n > _SUB else aa
    qs = np.quantile(s, [0.0, 0.01, 0.5, 0.99, 0.9999])
    out = {
        "tag": tag, "n": n, "absmax": absmax,
        "std": float(flat.std()), "mean": float(flat.mean()),
        "q_abs_min": float(qs[0]), "q_abs_p1": float(qs[1]),
        "q_abs_p50": float(qs[2]), "q_abs_p99": float(qs[3]),
        "q_abs_p9999": float(qs[4]),
        "frac_exact_zero": float(np.count_nonzero(flat == 0.0)) / n,
        "frac_lt_fp16_min_normal": float(np.mean(aa < FP16_MIN_NORMAL)),
        "frac_lt_fp16_min_sub": float(np.mean(aa < FP16_MIN_SUB)),
        "frac_gt_fp16_max": float(np.mean(aa > FP16_MAX)),
        "k_safe": float(np.floor(np.log2(FP16_MAX / absmax))) if absmax > 0 else 99.0,
        "k_ideal": float(np.floor(np.log2(32768.0 / absmax))) if absmax > 0 else 99.0,
        "step": CUR["step"],
    }
    if extra:
        out.update(extra)
    return out


def _note(rec):
    ROWS.append(rec)
    print("[ds] %s" % json.dumps(rec, ensure_ascii=False), flush=True)


def _cap(x, fn, tag):
    buf = RT.clone(x._gpu)
    out = T.Tensor._from_gpu(buf, requires_grad=True, _parents=(x,))

    def bw(g, grads):
        if CUR["step"] in TARGET:
            try:
                fn(g, tag)
            except Exception:                                    # noqa: BLE001
                traceback.print_exc()
        T._acc(grads, x, g)
    out._backward = bw
    return out


def _save(tag, arr, heads=2):
    p = os.path.join(SAMP, "%s_s%03d.npy" % (tag, CUR["step"]))
    np.save(p, np.ascontiguousarray(arr[:heads]))
    return p


def _dump(buf, tag, save=False):
    a = np.asarray(T.Tensor._from_gpu(buf, requires_grad=False).to_numpy(), np.float32)
    rec = _stats(a, tag, {"file": _save(tag, a)} if save else None)
    del a
    _note(rec)


# ---------------------------------------------------------------- SDPA（global 层）
_orig_scores = T._attn_scores


def _attn_scores_probe(q, k, scale):
    idx = CUR["sdpa"]
    CUR["sdpa"] = idx + 1
    want = (CUR["step"] in TARGET) and (idx in (0, 2))     # layer 3 / layer 11
    if not want:
        return _orig_scores(q, k, scale)
    rt = T._cuda()
    bh, s, dh = (int(x) for x in q.shape)
    buf = rt.sgemm_nt_strided_batched(q._gpu, k._gpu, bh, s, s, dh)
    if scale != 1.0:
        buf = rt.scale(buf, float(scale))
    out = T.Tensor._from_gpu(buf, requires_grad=bool(q.requires_grad or k.requires_grad),
                             _parents=(q, k))
    layer = 3 if idx == 0 else 11

    def bw(g, grads):
        if CUR["step"] in TARGET:
            try:
                _dump(g, "ds_sdpa_g%d" % layer, save=(layer == 3))
                if layer == 3:
                    for nm, t in (("k", k), ("q", q)):
                        _save("%s_sdpa_g%d" % (nm, layer),
                              np.asarray(T.Tensor._from_gpu(t._gpu, requires_grad=False)
                                         .to_numpy(), np.float32))
            except Exception:                                    # noqa: BLE001
                traceback.print_exc()
        dq = rt.sgemm_strided_batched(g, k._gpu, bh, s, s, dh)
        dk = rt.sgemm_tn_strided_batched(g, q._gpu, bh, s, dh, s)
        if scale != 1.0:
            dq = rt.scale(dq, float(scale))
            dk = rt.scale(dk, float(scale))
        if q.requires_grad:
            T._acc(grads, q, dq)
        if k.requires_grad:
            T._acc(grads, k, dk)

    if out.requires_grad:
        out._backward = bw
    return out


T._attn_scores = _attn_scores_probe

# ---------------------------------------------------------------- global 的 dp
_orig_sm = T._softmax_cuda
_N = [0]


def _softmax_probe(x, axis):
    shp = tuple(int(v) for v in x.shape)
    y = _orig_sm(x, axis)
    if (CUR["step"] in TARGET and len(shp) == 3 and shp[-1] >= 1024
            and _N[0] < 1):
        _N[0] += 1
        return _cap(y, lambda g, tag: _dump(g, tag), "dp_sdpa_g3")
    return y


T._softmax_cuda = _softmax_probe

# ---------------------------------------------------------------- CSA
_CSA_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10]
_orig_attn = C._attn


def _attn_probe(q, k, v, mask, row_mask, scale):
    idx = CUR["csa"]
    CUR["csa"] = idx + 1
    li = idx // 2
    layer = _CSA_LAYERS[li] if li < len(_CSA_LAYERS) else -1
    local = (idx % 2 == 1)
    want = (CUR["step"] in TARGET) and (layer in (0, 8))
    scores = (q @ k.swap_last2()) * float(scale)
    scores = scores + mask
    nm = "csa_l%d%s" % (layer, "_loc" if local else "")
    if want:
        scores = _cap(scores, lambda g, tag: _dump(g, tag), "ds_" + nm)
    p = T.functional.softmax(scores, axis=-1)
    if want and not local and layer == 0:
        p = _cap(p, lambda g, tag: _dump(g, tag), "dp_" + nm)
    out = p @ v
    if row_mask is not None:
        out = out * row_mask
    if CUR["step"] in TARGET and layer == 0 and not local:
        for tagn, t in (("k", k), ("q", q)):
            try:
                _save("%s_%s" % (tagn, nm),
                      np.asarray(T.Tensor._from_gpu(t._gpu, requires_grad=False)
                                 .to_numpy(), np.float32))
            except Exception:                                    # noqa: BLE001
                traceback.print_exc()
    return out


C._attn = _attn_probe

# ---------------------------------------------------------------- 步边界 / loss
_orig_sched_step = _optim.lr_scheduler.CosineLR.step


def _sched_step(self):
    STEP[0] += 1
    return _orig_sched_step(self)


_optim.lr_scheduler.CosineLR.step = _sched_step

_orig_fwd = TR.forward_loss


def _fwd(P, cfg, kinds, cos, sin, ids, labels, seg, **kw):
    CUR["step"] = STEP[0] + 1
    CUR["sdpa"] = 0
    CUR["csa"] = 0
    _N[0] = 0
    loss, meta = _orig_fwd(P, cfg, kinds, cos, sin, ids, labels, seg, **kw)
    if CUR["step"] in TARGET or CUR["step"] % 50 == 0:
        try:
            LOSS[CUR["step"]] = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
        except Exception:                                        # noqa: BLE001
            LOSS[CUR["step"]] = None
        print("[fwd] step %d loss %r" % (CUR["step"], LOSS[CUR["step"]]), flush=True)
    return loss, meta


TR.forward_loss = _fwd


def _report():
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"rows": ROWS, "loss": LOSS, "target": sorted(TARGET),
                   "fix_csa_mask": FIX_MASK}, f, ensure_ascii=False, indent=1)
    print("[probe] wrote %s (%d rows)" % (OUT, len(ROWS)))


atexit.register(_report)

argv = ["--data", os.path.join(WORK, "data2048_v96000"), "--vocab", "96000",
        "--seq", "2048", "--d", "1024", "--layers", "12", "--E", "4",
        "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
        "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
        "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
print("[probe] target steps = %s, total = %d, fix_mask=%r"
      % (sorted(TARGET), STEPS, FIX_MASK), flush=True)
TR.main(argv)
