# -*- coding: utf-8 -*-
"""把 ORV 每个 GEMM 入口的**调用点**抓出来：形状 → 谁调的、走 fp16 还是 fp32。

为什么需要它：ORV 的 `bmm` / `sgemm_strided_batched` 是 **fp32-only**（见 cuda.py:5851），
而 2D 的 `A @ B.T` 在 `--amp gemm` 下可能走 fp16 的 `gemm_f16_*_raw`。只看 kernel 名
（cublasGemmEx_fp16 vs cublasSgemmStridedBatched）无法判断哪个数学操作落在哪一侧，
所以这里按形状只记一次调用栈。

跑法（远端）：D:\\orv_pip_0962\\Scripts\\python.exe -X utf8 D:\\orv_bench\\probe_gemm_sites.py
"""
import atexit
import os
import sys
import traceback

WORK = r"D:\orv_bench"
os.chdir(WORK)
sys.path.insert(0, WORK)

import tgai_go.train as T                                        # noqa: E402
from tgai_go.train import v3_pretrain as TR                      # noqa: E402

RT = T._cuda()
SEEN = {}


def _shape(a):
    out = []
    for x in a:
        sh = getattr(x, "shape", None)
        if sh is not None:
            out.append(tuple(int(v) for v in sh))
        elif isinstance(x, int):
            out.append(x)
    return tuple(out)


def _wrap(name, fn):
    def f(*a, **kw):
        try:
            key = (name, _shape(a))
            if key not in SEEN:
                st = traceback.extract_stack()[-4:-1]
                SEEN[key] = ["%s:%d %s" % (os.path.basename(fr.filename), fr.lineno,
                                           fr.name) for fr in st]
        except Exception:                                        # noqa: BLE001
            pass
        return fn(*a, **kw)
    return f


NAMES = ("gemm_nt", "gemm_tn", "sgemm", "gemm_f16_nt_raw", "gemm_f16_nn_raw",
         "gemm_f16_tn_raw", "bmm", "sgemm_strided_batched", "gemm_ex",
         "sgemm_nt_strided_batched", "sgemm_tn_strided_batched")
for nm in NAMES:
    fn = getattr(RT, nm, None)
    if fn is None:
        print("[skip] no method %s" % nm)
        continue
    setattr(RT, nm, _wrap(nm, fn))


def _report():
    print("\n===== GEMM 入口 → 调用点（形状只记首现）=====")
    for (nm, sh), st in sorted(SEEN.items(), key=lambda kv: kv[0][0]):
        print("%-26s %-42s" % (nm, str(sh)[:42]))
        for s in st:
            print("      %s" % s)
    print("===== 合计 %d 种 (入口, 形状) 组合 =====" % len(SEEN))


atexit.register(_report)

argv = ["--data", os.path.join(WORK, "data2048_v96000"), "--vocab", "96000",
        "--seq", "2048", "--d", "1024", "--layers", "12", "--E", "4",
        "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
        "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
        "--steps", "5", "--log-every", "99", "--no-final-save"]
TR.main(argv)
