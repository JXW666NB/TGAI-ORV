# -*- coding: utf-8 -*-
"""枚举 ORV 每步所有「3-D / 批量 GEMM」调用点 → JSON（次数/FLOPs/栈/输入取值范围）。

为什么需要它：要判断「把 3-D 批量 GEMM 接进 fp16」值不值，先得知道**每步到底有哪些
(M,N,K,batch,op) 组合、各多少 FLOPs、浮点占比**。只按 kernel 名（cublasSgemmStridedBatched）
看是分不出「注意力 S×S」和「3-D @ 2-D 投影」的 —— 前者带宽受限、后者算力受限。

计数口径：只在「真正落到 cuBLAS 的那一层」计（_bmm_op / sgemm_strided_batched），
bmm→sgemm_strided_batched 这种父子关系用 DEPTH 挡住，不会重复计。

跑法（远端）：D:\\orv_pip_0962\\Scripts\\python.exe -X utf8 D:\\orv_bench\\probe_bmm_sites.py
产物：D:\\orv_bench\\bmm_sites.json
"""
import atexit
import collections
import json
import os
import sys
import traceback

WORK = r"D:\orv_bench"
OUT = os.path.join(WORK, "bmm_sites.json")
os.chdir(WORK)
sys.path.insert(0, WORK)

import numpy as np                                              # noqa: E402

import tgai_go.train as T                                       # noqa: E402
from tgai_go.train import v3_pretrain as TR                     # noqa: E402
from tgai_go.train import optim as _optim                       # noqa: E402

RT = T._cuda()
DEPTH = [0]
STEP = [0]

CNT = collections.Counter()
FLOP = collections.Counter()
PER_STEP = collections.OrderedDict()
CUR = collections.Counter()
INFO = {}
SAMPLE = {}
SAMPLE_TAG = set()


def _stk(n=6):
    fr = traceback.extract_stack()
    return ["%s:%d:%s" % (os.path.basename(f.filename), f.lineno, f.name)
            for f in fr[-(n + 2):-2]]


def _flops(*dims):
    v = 2
    for d in dims:
        v *= int(d)
    return v


def _guard(fn, entry, geomf):
    def f(*a, **kw):
        outer = (DEPTH[0] == 0)
        if outer:
            DEPTH[0] = 1
            try:
                g, fl = geomf(*a, **kw)
                if g is not None:
                    key = (entry, g)
                    CNT[key] += 1
                    FLOP[key] += fl
                    CUR[key] += 1
                    if key not in INFO:
                        INFO[key] = {"stack": _stk(), "flops": fl}
            except Exception:                                    # noqa: BLE001
                pass
        try:
            return fn(*a, **kw)
        finally:
            if outer:
                DEPTH[0] = 0
    return f


def _g_bmm(A, B):
    b, M, K = (int(x) for x in A.shape)
    _, K2, N = (int(x) for x in B.shape)
    return ("bmm b=%d M=%d N=%d K=%d" % (b, M, N, K), _flops(b, M, N, K))


def _g_ssb(A, B, batch, M, K, N):
    return ("ssb b=%d M=%d N=%d K=%d" % (batch, M, N, K),
            _flops(batch, M, N, K))


def _g_nt(A, B, batch, M, N, K):
    return ("ssbNT b=%d M=%d N=%d K=%d" % (batch, M, N, K),
            _flops(batch, M, N, K))


def _g_tn(A, B, batch, M, N, K):
    return ("ssbTN b=%d M=%d N=%d K=%d" % (batch, M, N, K),
            _flops(batch, M, N, K))


def _g_mm(A, B):
    sa = tuple(int(x) for x in A.shape)
    sb = tuple(int(x) for x in B.shape)
    M, K, N = sa[-2], sa[-1], sb[-1]
    lead = sa[:-2] if len(sa) > len(sb) else sb[:-2]
    b = 1
    for s in lead:
        b *= s
    return ("mmgen lead=%s M=%d N=%d K=%d" % ("x".join(str(x) for x in lead),
                                              M, N, K),
            _flops(b, M, N, K))


def _g_bmmop(op_a, op_b, Aptr, lda, stride_a, Bptr, ldb, stride_b,
             C, batch, M, N, K):
    tag = {0: "N", 1: "T"}.get(int(op_a), "?") + \
          {0: "N", 1: "T"}.get(int(op_b), "?")
    return ("bmmop %s b=%d M=%d N=%d K=%d" % (tag, batch, M, N, K),
            _flops(batch, M, N, K))


for _nm, _gf in [("bmm", _g_bmm), ("sgemm_strided_batched", _g_ssb),
                 ("sgemm_nt_strided_batched", _g_nt),
                 ("sgemm_tn_strided_batched", _g_tn),
                 ("matmul_general", _g_mm), ("_bmm_op", _g_bmmop)]:
    setattr(RT, _nm, _guard(getattr(RT, _nm), _nm, _gf))


# ---- 每步边界：CosineLR.step 每步恰好一次（v3_pretrain.py:1288）------
def _flush():
    STEP[0] += 1
    if CUR:
        PER_STEP[STEP[0]] = dict(CUR)
        CUR.clear()


_orig_sched_step = _optim.lr_scheduler.CosineLR.step


def _sched_step(self):
    _flush()
    return _orig_sched_step(self)


_optim.lr_scheduler.CosineLR.step = _sched_step


# ---- 取值采样：判 fp16 溢出用（第 2 步之后，每个 key 只取一次）--------
def _stats(buf):
    a = T.Tensor._from_gpu(buf, requires_grad=False).to_numpy()
    a = np.asarray(a, dtype=np.float64).ravel()
    return {"n": int(a.size), "min": float(a.min()), "max": float(a.max()),
            "absmax": float(np.abs(a).max()), "std": float(a.std()),
            "frac_gt_1e3": float(np.mean(np.abs(a) > 1e3)),
            "frac_gt_1e4": float(np.mean(np.abs(a) > 1e4))}


def _maybe_sample(entry, geom, bufs):
    key = (entry, geom)
    if STEP[0] < 2 or key in SAMPLE_TAG:
        return
    SAMPLE_TAG.add(key)
    try:
        SAMPLE["%s | %s" % key] = {
            "stack": INFO.get(key, {}).get("stack", []),
            "bufs": [_stats(b) for b in bufs]}
    except Exception as e:                                       # noqa: BLE001
        SAMPLE["%s | %s" % key] = {"err": repr(e)}


_orig_nt, _orig_ssb = RT.sgemm_nt_strided_batched, RT.sgemm_strided_batched
_orig_mm = RT.matmul_general


def _nt(*a, **kw):
    if DEPTH[0] == 0:
        A, B, batch, M, N, K = a[0], a[1], int(a[2]), int(a[3]), int(a[4]), int(a[5])
        _maybe_sample("sgemm_nt_strided_batched",
                      "ssbNT b=%d M=%d N=%d K=%d" % (batch, M, N, K), [A, B])
    return _orig_nt(*a, **kw)


def _ssb(*a, **kw):
    if DEPTH[0] == 0:
        A, B, batch, M, K, N = a[0], a[1], int(a[2]), int(a[3]), int(a[4]), int(a[5])
        _maybe_sample("sgemm_strided_batched",
                      "ssb b=%d M=%d N=%d K=%d" % (batch, M, N, K), [A, B])
    return _orig_ssb(*a, **kw)


def _mm(A, B):
    if DEPTH[0] == 0:
        sa = tuple(int(x) for x in A.shape)
        sb = tuple(int(x) for x in B.shape)
        M, K, N = sa[-2], sa[-1], sb[-1]
        lead = sa[:-2] if len(sa) > len(sb) else sb[:-2]
        b = 1
        for s in lead:
            b *= s
        _maybe_sample("matmul_general",
                      "mmgen lead=%s M=%d N=%d K=%d"
                      % ("x".join(str(x) for x in lead), M, N, K), [A, B])
    return _orig_mm(A, B)


RT.sgemm_nt_strided_batched = _nt
RT.sgemm_strided_batched = _ssb
RT.matmul_general = _mm


def _report():
    rows = []
    for (e, g), fl in FLOP.items():
        rows.append({"entry": e, "geom": g, "calls_total": CNT[(e, g)],
                     "flops_total": fl, "stack": INFO[(e, g)]["stack"]})
    rows.sort(key=lambda r: -r["flops_total"])
    steps = sorted(PER_STEP)
    out = {"steps": steps,
           "per_step": {str(s): {"%s | %s" % k: v for k, v in PER_STEP[s].items()}
                        for s in steps},
           "rows": rows,
           "samples": SAMPLE,
           "totals": {"calls_total": sum(CNT.values()),
                      "flops_total": sum(FLOP.values())}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[probe] wrote %s" % OUT)

    last = steps[-1] if steps else 0
    c = PER_STEP.get(last, {})
    tot_n = sum(c.values())
    tot_fl = sum(r["flops_total"] / len(steps) for r in rows)
    print("== step %d: %d 次调用, %.3f GFLOP" % (last, tot_n, tot_fl / 1e9))
    print("%-24s %-40s %6s %10s %8s" % ("entry", "geom", "n/step", "GFLOP/步", "ms@9.9"))
    for r in rows:
        n = c.get((r["entry"], r["geom"]), 0)
        if not n:
            continue
        fl = r["flops_total"] / len(steps)
        print("%-24s %-40s %6d %10.3f %8.2f"
              % (r["entry"], r["geom"][:40], n, fl / 1e9, fl / 1e9 / 9.9))
    print("\n== 采样 ==")
    print(json.dumps(SAMPLE, ensure_ascii=False, indent=1)[:8000])


atexit.register(_report)

argv = ["--data", os.path.join(WORK, "data2048_v96000"), "--vocab", "96000",
        "--seq", "2048", "--d", "1024", "--layers", "12", "--E", "4",
        "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
        "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
        "--steps", "3", "--log-every", "99", "--no-final-save"]
TR.main(argv)
