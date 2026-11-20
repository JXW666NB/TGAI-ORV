# -*- coding: utf-8 -*-
"""取证：fp32 批量 GEMM 的 20.3 ms 到底是前向还是反向？

做法：包 `_bmm_op`，记录 (batch,M,N,K) 与**调用栈**（区分是哪个调用方、前向还是反向），
按 (形状, 栈顶3帧) 聚合次数。不修改任何源码，退出即失效。
"""
import atexit
import json
import sys
import traceback
from collections import Counter

import tgai_go.train as T
from tgai_go.train import v3_pretrain as TR

rt = T._cuda()
hits = Counter()
_orig = rt._bmm_op


def wrapped(op_a, op_b, Aptr, lda, sa, Bptr, ldb, sb, C, batch, M, N, K):
    st = traceback.extract_stack(limit=6)
    stack = []
    for fr in reversed(st[:-1]):
        if "_bmm_op" in fr.filename + str(fr.lineno):
            continue
        f = fr.filename.split("\\")[-1]
        if f in ("cuda.py", "__init__.py", "v3_csa.py", "v3_moe.py"):
            stack.append("%s:%d" % (f, fr.lineno))
        if len(stack) >= 3:
            break
    hits[(int(batch), int(M), int(N), int(K), tuple(stack))] += 1
    return _orig(op_a, op_b, Aptr, lda, sa, Bptr, ldb, sb, C, batch, M, N, K)


rt._bmm_op = wrapped


def _report():
    rows = []
    for (b, m, n, k, stack), c in hits.most_common(12):
        rows.append({"b": b, "M": m, "N": n, "K": k, "calls": c,
                     "at": " <- ".join(stack)})
    print("[bmm] " + json.dumps(rows, ensure_ascii=False, indent=1))


atexit.register(_report)

if __name__ == "__main__":
    TR.main(sys.argv[1:])
