#!/usr/bin/env bash
# 验证 pinned 大块传输是否真的把带宽拉起来（对 3.16 GiB）
cat > /root/orv_rent/bench_pinned.py <<'PY'
import time
import numpy as np
import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
N = 848_645_257
gb = N * 4 / float(1 << 30)
buf = rt.empty((N,), dtype=np.float32)
rt.h2d(np.zeros(N, dtype=np.float32))

# 对比 1：旧的逐次 d2h/h2d（可分页，每次新分配）
t0 = time.perf_counter(); a = rt.d2h(buf); t1 = time.perf_counter()
b = rt.h2d(a); t2 = time.perf_counter()
print("可分页  d2h %5.2f GiB/s   h2d %5.2f GiB/s" % (gb/(t1-t0), gb/(t2-t1)))
del a, b

# 对比 2：pinned 复用缓冲 + 一次整块拷贝
ph = rt.host_pinned(N * 4, slot="bench")
print("pinned 容量 %.2f GiB" % (ph.nbytes / float(1 << 30)))
t0 = time.perf_counter(); rt.d2h_into(buf, ph.ptr); t1 = time.perf_counter()
rt.h2d_into(buf, ph.ptr); t2 = time.perf_counter()
print("pinned  d2h %5.2f GiB/s   h2d %5.2f GiB/s   往返 %5.3f s"
      % (gb/(t1-t0), gb/(t2-t1), (t2-t0)))

# 再来一轮（排除首次分配的抖动）
t0 = time.perf_counter(); rt.d2h_into(buf, ph.ptr); t1 = time.perf_counter()
rt.h2d_into(buf, ph.ptr); t2 = time.perf_counter()
print("pinned2 d2h %5.2f GiB/s   h2d %5.2f GiB/s   往返 %5.3f s"
      % (gb/(t1-t0), gb/(t2-t1), (t2-t0)))
PY
cd /root/orv_rent && /root/miniconda3/bin/python bench_pinned.py 2>&1 | tail -8
