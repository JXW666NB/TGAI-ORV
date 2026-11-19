# -*- coding: utf-8 -*-
"""端到端 A/B：把「3-D @ 2-D（batch=1）」的投影 GEMM 绕回**已有**的 2-D fp16 路径。

## 为什么这是最大的一块

探针（probe_bmm_sites.py）实测：每步 369 次 cublasSgemmStridedBatched 里，有
60 次是 `lead=1 M=2048 N=1024 K=1024`（12 层 × 5 个投影），另有 9+9+18+18 次同类
小投影 —— 它们 batch=1，本来就没必要走 strided-batched，却因为输入是 3-D 而掉了
`_matmul_cuda_nd` 分支，于是完全绕开了 `--amp gemm` 已有的 fp16 路径。

本脚本**不改源码**，只 monkeypatch 三个入口，把 batch==1 的调用换成 2-D 等价形式
（`rt.sgemm` / `rt.gemm_nt` / `rt.gemm_tn`），再测步时与 loss。

用法（远端）：
    ... python.exe -X utf8 D:\\orv_bench\\ab_route_2d.py off|on|ab [steps]
      off —— 完全原样（基线）
      on  —— 全程走 2-D 路径
      ab  —— 同一进程内前 8 步关、之后开（最公平的步时对比）
"""
import os
import re
import statistics
import sys

WORK = r"D:\orv_bench"
os.chdir(WORK)
sys.path.insert(0, WORK)

import tgai_go.train as T                                       # noqa: E402
from tgai_go.train import v3_pretrain as TR                     # noqa: E402
from tgai_go.train import optim as _optim                       # noqa: E402

RT = T._cuda()
EN = [False]
MODE = (sys.argv[1] if len(sys.argv) > 1 else "ab").lower()
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
FLIP_AT = 8
STAT = {"hit_mm": 0, "hit_nt": 0, "hit_tn": 0}
NOTN = (len(sys.argv) > 3 and sys.argv[3].lower() == "notn")

_orig_mm = RT.matmul_general
_orig_nt = RT.sgemm_nt_strided_batched
_orig_tn = RT.sgemm_tn_strided_batched
_orig_sgemm = RT.sgemm
_orig_gnt = RT.gemm_nt
_orig_gtn = RT.gemm_tn


def _flat(buf):
    """[1,M,K] -> [M,K]（元数据 reshape，不拷贝）；其它形状原样返回。"""
    if len(buf.shape) == 3 and int(buf.shape[0]) == 1:
        return RT.reshape(buf, (int(buf.shape[1]), int(buf.shape[2])))
    return buf


def mm(A, B):
    if EN[0] and len(A.shape) == 3 and int(A.shape[0]) == 1 and len(B.shape) == 2:
        M, K = int(A.shape[1]), int(A.shape[2])
        N = int(B.shape[1])
        A2 = RT.reshape(A, (M, K))
        C2 = _orig_sgemm(A2, B)                  # 2-D 路径（含 _sgemm 的 fp16 分支）
        STAT["hit_mm"] += 1
        return RT.reshape(C2, (1, M, N))
    return _orig_mm(A, B)


def nt(A, B, batch, M, N, K):
    if EN[0] and int(batch) == 1:
        A2, B2 = _flat(A), _flat(B)
        if len(A2.shape) == 2 and len(B2.shape) == 2:
            STAT["hit_nt"] += 1
            return _orig_gnt(A2, B2)             # C[M,N] = A[M,K] @ B[N,K]^T
    return _orig_nt(A, B, batch, M, N, K)


def tn(A, B, batch, M, N, K):
    if EN[0] and int(batch) == 1 and not NOTN:
        A2, B2 = _flat(A), _flat(B)
        if len(A2.shape) == 2 and len(B2.shape) == 2:
            STAT["hit_tn"] += 1
            return _orig_gtn(A2, B2)             # C[M,N] = A[K,M]^T @ B[K,N]
    return _orig_tn(A, B, batch, M, N, K)


RT.matmul_general = mm
RT.sgemm_nt_strided_batched = nt
RT.sgemm_tn_strided_batched = tn

if MODE == "on":
    EN[0] = True

_orig_sched = _optim.lr_scheduler.CosineLR.step
_N = [0]


def _sched(self):
    i = _N[0]
    if MODE == "ab" and i == FLIP_AT:
        EN[0] = True
        print("@@@ ROUTE ON at step %d (hits so far mm=%d nt=%d tn=%d)"
              % (i, STAT["hit_mm"], STAT["hit_nt"], STAT["hit_tn"]), flush=True)
    _N[0] = i + 1
    return _orig_sched(self)


_optim.lr_scheduler.CosineLR.step = _sched

argv = ["--data", os.path.join(WORK, "data2048_v96000"), "--vocab", "96000",
        "--seq", "2048", "--d", "1024", "--layers", "12", "--E", "4",
        "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
        "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
        "--steps", str(STEPS), "--log-every", "1", "--no-final-save"]

print("@@@ MODE=%s steps=%d" % (MODE, STEPS), flush=True)
TR.main(argv)
print("@@@ hits mm=%d nt=%d tn=%d" % (STAT["hit_mm"], STAT["hit_nt"],
                                      STAT["hit_tn"]), flush=True)
