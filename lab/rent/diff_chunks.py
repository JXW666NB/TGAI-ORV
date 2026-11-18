# -*- coding: utf-8 -*-
"""差分定位流水竞态：同一份梯度，K=1（串行，正确）vs K=8（流水）比较结果。

两进程各构造 321 个梯度（总和 3.16 GiB 量级的 1/16，便于快速反复），
先跑 K=1 记下结果，恢复梯度再跑 K=8，逐段比较，打印出错的偏移范围。
"""
import os

import numpy as np

import tgai_go.train as T
from tgai_go.train import dist as D

T.set_default_device("cuda")
rt = T._cuda()
dp = D.DataParallel()

N = 848_645_257 // 16          # 缩小 16 倍，便于快速对比
NPAR = 64
chunk = N // NPAR

params = []
for i in range(NPAR):
    n = chunk if i < NPAR - 1 else N - chunk * (NPAR - 1)
    p = T.tensor(np.zeros(n, np.float32), requires_grad=True)
    # 每个 rank 给自己不同的梯度值，便于看出"少了平均"还是"读到旧值"
    p.grad = T.tensor(np.full(n, float(dp.rank + 1), np.float32))
    params.append(p)


def read_grads():
    out = []
    for p in params:
        out.append(np.array(rt.d2h(p.grad._gpu), dtype=np.float32))
    return np.concatenate(out)


def reset():
    for p in params:
        p.grad = T.tensor(np.full(int(p._gpu.n), float(dp.rank + 1), np.float32))


os.environ["TG_DP_CHUNKS"] = "1"
reset()
dp.all_reduce_grads(params)
a = read_grads()

os.environ["TG_DP_CHUNKS"] = "8"
reset()
dp.all_reduce_grads(params)
b = read_grads()

# 期望：每个 rank 看到都是 2.0（(1+1)/2? 不对：rank0=1,rank1=2 ⇒ 平均 1.5）
exp = 1.5 * 1.0
d = np.abs(a - b)
print("rank=%d  K=1 首值 %.3f 末值 %.3f  |  K=8 首值 %.3f 末值 %.3f"
      % (dp.rank, a[0], a[-1], b[0], b[-1]), flush=True)
print("  K=1 期望值 %.3f，最大偏差 %.3e" % (exp, float(np.abs(a - exp).max())), flush=True)
print("  K=8 最大偏差 %.3e；两者最大差 %.3e" % (float(np.abs(b - exp).max()),
                                         float(d.max())), flush=True)
bad = np.nonzero(d > 1e-6)[0]
if bad.size:
    print("  不一致元素 %d 个，范围 [%d, %d]，分块边界=%s"
          % (bad.size, int(bad[0]), int(bad[-1]),
             [int(N * i // 8) for i in range(9)]), flush=True)
else:
    print("  两者完全一致", flush=True)
