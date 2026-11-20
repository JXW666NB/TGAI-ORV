# -*- coding: utf-8 -*-
"""数 kernel：在同一进程里跑 v3_pretrain，包住 rt.launch 统计每个内核的发射次数。

用法： python probe_kernels.py gemm|off
"""
import collections
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "off"

sys.argv = [
    "v3_pretrain",
    "--data", "/root/bench/data2048_v96000", "--vocab", "96000",
    "--seq", "512", "--d", "1024", "--layers", "12", "--E", "4",
    "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
    "--opt", "int8gpu", "--amp", MODE, "--batch", "1",
    "--steps", "2", "--log-every", "99", "--no-final-save",
]

import tgai_go.train as T                                    # noqa: E402

rt = T._cuda()
orig = rt.launch
cnt = collections.Counter()


def counting(name, *a, **k):
    cnt[name] += 1
    return orig(name, *a, **k)


rt.launch = counting

from tgai_go.train import v3_pretrain as VP                  # noqa: E402

try:
    VP.main()
except SystemExit:
    pass

total = sum(cnt.values())
print("\n[probe] amp=%s  总发射 %d 次（2 步）=> 每步 %.0f 次"
      % (MODE, total, total / 2.0))
print("[probe] 发射次数最多的 14 个内核：")
for n, c in cnt.most_common(14):
    print("   %-34s %6d" % (n, c))
print("[probe] 不同内核种类: %d" % len(cnt))
