# -*- coding: utf-8 -*-
"""kernel 级 GPU 归因（CUDA event，不 sync 逐 kernel）。

## 为什么这么写

历史坑：以前所有 kernel 归因都是「调用前后各 rt.sync()」，于是**探测自身的开销**
被算进 kernel —— `transpose2d` 被量成 76 ms/步，可端到端替换掉它步时一点没变。
自检脚本 `selftest_event.py` 里复现了这个放大（1 M 元素的 add：sync 包裹 34.5 µs
vs event 20.6 µs，放大 1.7×；越小的 kernel 放大越狠）。

这里的做法：
  * 每个**叶子** GPU 调用前后各记一个 **event**（`cuEventRecord` 只往流里塞一个
    时间戳写操作，不阻塞主机、不打断发射流水）；
  * 一步结束（训练器自己的 `rt.sync()`）之后，**批量**读 `cuEventElapsedTime`；
  * event 对象池化复用，避免数千次/步 的 `cuEventCreate` 成本；
  * 只对叶子打点：`rt.launch`（NVRTC 内核）和 cuBLAS / async memcpy / memset 的
    ctypes 函数本身。上层 Python 方法（`_sgemm`、`bmm` 等）**不**打点，否则
    「cast 内核」会被父区间重复计算一次；
  * **相邻叶子之间的空档也算**（end_i → start_{i+1}）：这块正是"GPU 在等主机"
    或"有未打点的 GPU 操作"，不算出来就无法解释 步时 与 Σ内核 的差；
  * 另外统计**主机侧同步/分配类调用**的次数与主机耗时（cuMemAlloc / cuMemFree /
    cuCtxSynchronize / cuStreamSynchronize / 同步 memcpy）——它们是"整设备同步"
    型停顿的嫌疑人，且不产生可计时的 GPU 区间。

## 产出

一步一个 JSON（默认 `D:\\orv_bench\\prof_kernels.json`），再由
`run_prof_seq.py` 汇总成表 A/B/C/D。

用法：
    python prof_event_kernels.py --data D:\\orv_bench\\data2048_v96000 --seq 2048 ...
"""
import atexit
import ctypes
import json
import os
import sys
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_pretrain as TR

RT = T._cuda()

SKIP = int(os.environ.get("PROF_SKIP", "3"))
OUT = os.environ.get("PROF_OUT", r"D:\orv_bench\prof_kernels.json")
TAG = os.environ.get("PROF_TAG", "")

_C_INT = ctypes.c_int
_U64 = ctypes.c_uint64

# ── 状态 ──────────────────────────────────────────────────────────────────
_rec = [False]                 # 是否处于计时窗口
STEP = [-1]                    # forward_loss 调用序号（= 步号）
PHASE = ["other"]
FREE = []                      # 空闲 event 池
PEND = []                      # [(ev0, ev1, label, aux, phase)]  待读
PPEND = []                     # [(ev0, ev1, phase)]              阶段段
OPS = {}                       # (label, aux) -> [n, ms, {phase: ms}]
PHS = {}                       # phase -> [n, ms]
GAPS = {}                      # (prev_label, next_label) -> [n, ms]
SAMP = {}                      # label -> {aux: 描述}
HOST = {}                      # (主机侧调用名, phase) -> [n, ms]
HOSTL = {}                     # 叶子 label -> [n, CPU ms]（发射一次要多少 CPU）
POL = {}                       # phase -> [dhits, dmisses, dtrims]
_prev_pool = [None]
STEP_WALL = []
_LAST_T = [None]
NMEAS = [0]

_fast = {}
_np = [0]                      # 每步叶子数（增量）


def _init_fast():
    _fast["rec"] = RT.cu.cuEventRecord
    _fast["stream"] = RT._stream


def _take():
    if FREE:
        return FREE.pop()
    ev = ctypes.c_void_p()
    RT._chk(RT.cu.cuEventCreate(ctypes.byref(ev), ctypes.c_uint(0)), "cuEventCreate")
    return ev


def _mark():
    ev = _take()
    if _fast["rec"](ev, _fast["stream"]) != 0:
        raise RuntimeError("cuEventRecord failed")
    return ev


def _give(ev):
    FREE.append(ev)


def _describe(grid, block, shared, ints, nptr):
    return {"grid": [int(x) for x in grid], "block": [int(x) for x in block],
            "shared": int(shared), "ints": list(ints[:6]), "nptr": int(nptr)}


def _commit(label, aux, phase, ms):
    k = (label, aux)
    st = OPS.get(k)
    if st is None:
        OPS[k] = st = [0, 0.0, {}]
    st[0] += 1
    st[1] += ms
    st[2][phase] = st[2].get(phase, 0.0) + ms


def _gap(prev, nxt, ms):
    k = (prev, nxt)
    st = GAPS.get(k)
    if st is None:
        GAPS[k] = st = [0, 0.0]
    st[0] += 1
    st[1] += ms


def _pool_delta(tag):
    """把「上次快照以来」的池统计增量记到 tag 阶段上。

    用来回答：这个阶段的显存分配到底有多碎（misses）以及被 trim 了多少次。
    """
    try:
        st = RT.pool_stats()
    except Exception:
        return
    cur = (int(st["hits"]), int(st["misses"]), int(st["trims"]))
    p = _prev_pool[0]
    _prev_pool[0] = cur
    if p is None or tag is None:
        return
    rec = POL.get(tag)
    if rec is None:
        POL[tag] = rec = [0, 0, 0]
    for i in range(3):
        rec[i] += cur[i] - p[i]


def _drain(final=False):
    """把待读的 event 区间批量读成毫秒。没跑完就整批留着下次再读。

    ★ 必须"先读齐、再提交"：中途失败时不能已经 commit 了一半又把同一批
      留到下次，否则会重复计入（第一版踩过：n_leaf_per_step 直接翻倍）。
    """
    if not PEND and not PPEND:
        return
    if PEND:
        vals = []
        ok = True
        for ev0, ev1, label, aux, phase in PEND:
            try:
                vals.append(RT.event_elapsed_ms(ev0, ev1))
            except Exception:
                ok = False
                break
        if not ok and not final:
            return
        if ok:
            gvals = []
            for i in range(1, len(PEND)):
                try:
                    gvals.append(RT.event_elapsed_ms(PEND[i - 1][1], PEND[i][0]))
                except Exception:
                    gvals = []
                    break
            for i, (ev0, ev1, label, aux, phase) in enumerate(PEND):
                _commit(label, aux, phase, vals[i])
            for i, g in enumerate(gvals, 1):
                _gap(PEND[i - 1][2], PEND[i][2], g)
        try:
            for ev0, ev1, label, aux, phase in PEND:
                _give(ev0)
                _give(ev1)
        finally:
            del PEND[:]

    if PPEND:
        pvals = []
        pok = True
        for ev0, ev1, phase in PPEND:
            try:
                pvals.append(RT.event_elapsed_ms(ev0, ev1))
            except Exception:
                pok = False
                break
        if pok:
            for i, (ev0, ev1, phase) in enumerate(PPEND):
                st = PHS.get(phase)
                if st is None:
                    PHS[phase] = st = [0, 0.0]
                st[0] += 1
                st[1] += pvals[i]
        try:
            for ev0, ev1, phase in PPEND:
                _give(ev0)
                _give(ev1)
        finally:
            del PPEND[:]


# ── 打点：NVRTC 内核 ─────────────────────────────────────────────────────
_orig_launch = RT.launch


def _launch(name, grid, block, args, shared=0, module=None):
    if not _rec[0]:
        return _orig_launch(name, grid, block, args, shared, module)
    t0 = time.perf_counter()
    ints = []
    nptr = 0
    for val, typ in args:
        if typ is _C_INT:
            ints.append(int(val))
        elif typ is _U64:
            nptr += 1
    if ints:
        aux = tuple(ints[:3])
    else:
        aux = (int(grid[0]) * int(block[0]),)
    ph = PHASE[0]
    e0 = _mark()
    out = _orig_launch(name, grid, block, args, shared, module)
    e1 = _mark()
    PEND.append((e0, e1, name, aux, ph))
    st = HOSTL.get(name)
    if st is None:
        HOSTL[name] = st = [0, 0.0]
    st[0] += 1
    st[1] += (time.perf_counter() - t0) * 1000.0
    d = SAMP.get(name)
    if d is None:
        SAMP[name] = {aux: _describe(grid, block, shared, ints, nptr)}
    elif aux not in d and len(d) < 12:
        d[aux] = _describe(grid, block, shared, ints, nptr)
    return out


RT.launch = _launch


# ── 打点：cuBLAS / async memcpy / memset（叶子）────────────────────────────
class _Leaf(object):
    """包装一个 ctypes 函数：前后各记一个 event。

    `argtypes` / `restype` 转发给内层函数 —— 这是必须的：cuda.py 里
    `_bind_gemm_ex()` / `sgemm_strided_batched()` 会**在运行中**给
    `self.cublas.xxx` 设 argtypes，若被我们包住而不转发，内层函数就永远没有
    argtypes，64 位指针会被 ctypes 当 32 位 int 截断（必崩/必错）。
    """

    __slots__ = ("fn", "label", "shape")

    def __init__(self, fn, label, shape):
        self.fn = fn
        self.label = label
        self.shape = shape

    def __call__(self, *a):
        if not _rec[0]:
            return self.fn(*a)
        t0 = time.perf_counter()
        ph = PHASE[0]
        e0 = _mark()
        rc = self.fn(*a)
        e1 = _mark()
        PEND.append((e0, e1, self.label, self.shape(a), ph))
        st = HOSTL.get(self.label)
        if st is None:
            HOSTL[self.label] = st = [0, 0.0]
        st[0] += 1
        st[1] += (time.perf_counter() - t0) * 1000.0
        return rc

    @property
    def argtypes(self):
        return self.fn.argtypes

    @argtypes.setter
    def argtypes(self, v):
        self.fn.argtypes = v

    @property
    def restype(self):
        return self.fn.restype

    @restype.setter
    def restype(self, v):
        self.fn.restype = v


class _Count(object):
    """只统计**主机侧**次数与主机耗时（这些调用会同步整设备/分配显存，
    本身不产生可 event 计时的 GPU 区间，但会造成停顿）。按阶段分开记。"""

    __slots__ = ("fn", "name")

    def __init__(self, fn, name):
        self.fn = fn
        self.name = name

    def __call__(self, *a):
        if not _rec[0]:
            return self.fn(*a)
        t0 = time.perf_counter()
        rc = self.fn(*a)
        dt = (time.perf_counter() - t0) * 1000.0
        k = (self.name, PHASE[0])
        st = HOST.get(k)
        if st is None:
            HOST[k] = st = [0, 0.0]
        st[0] += 1
        st[1] += dt
        return rc

    @property
    def argtypes(self):
        return self.fn.argtypes

    @argtypes.setter
    def argtypes(self, v):
        self.fn.argtypes = v

    @property
    def restype(self):
        return self.fn.restype

    @restype.setter
    def restype(self, v):
        self.fn.restype = v


_CUDA_R_16F = 2
_CUDA_R_16BF = 14


def _gemm_shape(a):
    try:
        return (int(a[3]), int(a[4]), int(a[5]))
    except Exception:
        return (0, 0, 0)


def _gemm_shape_batched(a):
    try:
        return (int(a[3]), int(a[4]), int(a[5]), int(a[-1]))
    except Exception:
        return (0, 0, 0, 0)


def _gemmex_label(a):
    try:
        t = int(a[8])
    except Exception:
        return "cublasGemmEx"
    if t == _CUDA_R_16F:
        return "cublasGemmEx_fp16"
    if t == _CUDA_R_16BF:
        return "cublasGemmEx_bf16"
    return "cublasGemmEx_other"


def _bytes_shape(a):
    """memcpy/memset：取字节数。所有调用点都用 ctypes.c_size_t 传字节/元素数。

    > 2^40（1 TiB）视为读错参数（位置记错 / argtypes 不对），记 0 而不是把
      荒谬数字混进表里 —— 静默的错数字比没有数字更坏。
    """
    for x in a:
        if isinstance(x, ctypes.c_size_t):
            v = int(x.value)
            return (v,) if v <= (1 << 40) else (0,)
    for x in reversed(a):
        if isinstance(x, int) and 0 <= x <= (1 << 40):
            return (int(x),)
    return (0,)


def _install_leaf_wrappers():
    cb = RT.cublas

    class _GemmEx(_Leaf):
        __slots__ = ()

        def __call__(self, *a):
            self.label = _gemmex_label(a)
            return _Leaf.__call__(self, *a)

    cb.cublasGemmEx = _GemmEx(cb.cublasGemmEx, "cublasGemmEx", _gemm_shape)
    cb.cublasSgemm_v2 = _Leaf(cb.cublasSgemm_v2, "cublasSgemm_fp32", _gemm_shape)
    cb.cublasSgemmStridedBatched = _Leaf(
        cb.cublasSgemmStridedBatched, "cublasSgemmStridedBatched", _gemm_shape_batched)
    grouped = getattr(cb, "cublasSgemmGroupedBatched", None)
    if grouped is not None:
        cb.cublasSgemmGroupedBatched = _Leaf(
            grouped, "cublasSgemmGroupedBatched", _gemm_shape)

    cu = RT.cu
    for nm in ("cuMemsetD32Async", "cuMemcpyHtoDAsync", "cuMemcpyDtoDAsync",
               "cuMemcpyDtoHAsync"):
        fn = getattr(cu, nm, None)
        if fn is not None:
            setattr(cu, nm, _Leaf(fn, "mem:" + nm, _bytes_shape))

    # 主机侧同步/分配类：只数次数 + 主机耗时
    # cuMemGetInfo 也要数：`_alloc` 在"接近显存上限"时每次未命中都会问一次驱动
    # （代码注释里实测 77 µs/次），seq2048 把显存顶到 23.7/32 GiB 时极易触发。
    for nm in ("cuMemAlloc", "cuMemFree", "cuCtxSynchronize", "cuStreamSynchronize",
               "cuMemcpyDtoD", "cuMemcpyHtoD", "cuMemcpyDtoH",
               "cuMemGetInfo_v2", "cuMemGetInfo"):
        fn = getattr(cu, nm, None)
        if fn is not None:
            setattr(cu, nm, _Count(fn, nm))


# ── 打点：四个阶段 ───────────────────────────────────────────────────────
_orig_forward_loss = TR.forward_loss
_orig_bwd = T.Tensor.backward
_orig_clip = T.clip_grad_norm_


def _forward_loss(*a, **kw):
    STEP[0] += 1
    t = time.perf_counter()
    if _LAST_T[0] is not None:
        STEP_WALL.append(t - _LAST_T[0])
    _LAST_T[0] = t
    if STEP[0] < SKIP:
        return _orig_forward_loss(*a, **kw)
    if not _rec[0]:
        _rec[0] = True
        _init_fast()
        _install_leaf_wrappers()
    _pool_delta("other")
    _pool_delta(None)
    e0 = _mark()
    PHASE[0] = "forward"
    out = _orig_forward_loss(*a, **kw)
    e1 = _mark()
    PPEND.append((e0, e1, "forward"))
    _pool_delta("forward")
    _pool_delta(None)
    PHASE[0] = "other"
    NMEAS[0] += 1
    return out


def _backward(self, *a, **kw):
    if not _rec[0]:
        return _orig_bwd(self, *a, **kw)
    _pool_delta("other")
    _pool_delta(None)
    e0 = _mark()
    PHASE[0] = "backward"
    out = _orig_bwd(self, *a, **kw)
    e1 = _mark()
    PPEND.append((e0, e1, "backward"))
    _pool_delta("backward")
    _pool_delta(None)
    PHASE[0] = "other"
    return out


def _clip(*a, **kw):
    if not _rec[0]:
        return _orig_clip(*a, **kw)
    _pool_delta("other")
    _pool_delta(None)
    e0 = _mark()
    PHASE[0] = "clip"
    out = _orig_clip(*a, **kw)
    e1 = _mark()
    PPEND.append((e0, e1, "clip"))
    _pool_delta("clip")
    _pool_delta(None)
    PHASE[0] = "other"
    return out


def _install_opt_hook():
    """优化器阶段：--opt int8gpu ⇒ Int8AdamWGpu.step（一次发射覆盖全部参数）。"""
    try:
        from tgai_go.train.int8_adamw_gpu import Int8AdamWGpu
    except Exception:
        return False
    orig = Int8AdamWGpu.step

    def step(self):
        if not _rec[0]:
            return orig(self)
        _pool_delta("other")
        _pool_delta(None)
        e0 = _mark()
        PHASE[0] = "optimizer"
        out = orig(self)
        e1 = _mark()
        PPEND.append((e0, e1, "optimizer"))
        _pool_delta("optimizer")
        _pool_delta(None)
        PHASE[0] = "other"
        return out

    Int8AdamWGpu.step = step
    return True


_orig_sync = RT.sync


def _sync():
    _orig_sync()
    if _rec[0]:
        _drain()


TR.forward_loss = _forward_loss
T.Tensor.backward = _backward
T.clip_grad_norm_ = _clip
RT.sync = _sync
_install_opt_hook()


# ── 收尾：出 JSON ────────────────────────────────────────────────────────
def _report():
    try:
        _drain(final=True)
    except Exception as ex:                     # noqa: BLE001
        print("[prof] drain failed: %r" % (ex,), flush=True)
    n = max(1, NMEAS[0])
    ops = []
    for (label, aux), (cnt, ms, byph) in OPS.items():
        ops.append({"label": label, "aux": list(aux),
                    "n_per_step": cnt / float(n),
                    "ms_per_step": ms / float(n),
                    "phase": {k: v / float(n) for k, v in byph.items()}})
    ops.sort(key=lambda r: -r["ms_per_step"])
    ph = {}
    for k, (cnt, ms) in PHS.items():
        ph[k] = {"n_per_step": cnt / float(n), "ms_per_step": ms / float(n)}
    gaps = [{"prev": k[0], "next": k[1], "n_per_step": v[0] / float(n),
             "ms_per_step": v[1] / float(n)} for k, v in GAPS.items()]
    gaps.sort(key=lambda r: -r["ms_per_step"])
    host = {}
    for nm, phv in list(HOST.items()):
        hname, hphase = nm
        dd = host.setdefault(hname, {"n_per_step": 0.0, "ms_per_step": 0.0,
                                     "phase": {}})
        dd["n_per_step"] += phv[0] / float(n)
        dd["ms_per_step"] += phv[1] / float(n)
        dd["phase"][hphase] = dd["phase"].get(hphase, 0.0) + phv[1] / float(n)
    hostl = {k: {"n_per_step": v[0] / float(n), "host_ms_per_step": v[1] / float(n)}
             for k, v in HOSTL.items()}
    pol = {str(k): [v[0] / float(n), v[1] / float(n), v[2] / float(n)]
           for k, v in POL.items()}
    wall = STEP_WALL[SKIP:] if len(STEP_WALL) > SKIP else STEP_WALL
    try:
        pstats = {k: (int(v) if isinstance(v, (int, float)) else v)
                  for k, v in RT.pool_stats().items()}
    except Exception:
        pstats = {}
    res = {
        "tag": TAG,
        "measured_steps": NMEAS[0],
        "skipped_steps": SKIP,
        "step_s_wall_mean": float(np.mean(wall)) if wall else None,
        "step_s_wall_all": [float(x) for x in wall],
        "pool_stats": pstats,
        "sum_leaf_ms_per_step": float(sum(r["ms_per_step"] for r in ops)),
        "sum_gap_ms_per_step": float(sum(r["ms_per_step"] for r in gaps)),
        "phases": ph,
        "n_label_shape": len(ops),
        "n_leaf_per_step": float(sum(r["n_per_step"] for r in ops)),
        "ops": ops,
        "gaps": gaps[:60],
        "host": host,
        "host_leaf": hostl,
        "pool": pol,
        "samples": {k: {str(a): d for a, d in v.items()} for k, v in SAMP.items()},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[prof] wrote %s  measured=%d  leaf=%.1f ms  gap=%.1f ms  wall=%.3f s"
          % (OUT, NMEAS[0], res["sum_leaf_ms_per_step"],
             res["sum_gap_ms_per_step"], res["step_s_wall_mean"] or 0.0), flush=True)


atexit.register(_report)

if __name__ == "__main__":
    TR.main(sys.argv[1:])
