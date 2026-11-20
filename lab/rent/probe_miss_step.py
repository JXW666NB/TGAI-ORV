# -*- coding: utf-8 -*-
"""按步快照 pool misses：判断未命中是冷启动还是稳态churn。"""
import builtins, sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "off"
sys.argv = ["v3_pretrain", "--data", "/root/bench/data2048_v96000", "--vocab", "96000",
            "--seq", "512", "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "int8gpu", "--amp", MODE, "--batch", "1",
            "--steps", "6", "--log-every", "1", "--no-final-save"]
import tgai_go.train as T
rt = T._cuda()
snaps = []
_orig = builtins.print
def spy(*a, **k):
    _orig(*a, **k)
    try:
        s = a[0] if a else ""
        if isinstance(s, str) and s.startswith("step"):
            st = rt.pool_stats()
            snaps.append((s.split()[1], int(st.get("misses", 0)), int(st.get("cached_bytes", 0)), int(st.get("live_bytes", 0))))
    except Exception:
        pass
builtins.print = spy
from tgai_go.train import v3_pretrain as VP
try: VP.main()
except SystemExit: pass
builtins.print = _orig
print("\n[miss-probe] amp=%s  每步快照（step / 累计misses / cached GiB / live GiB）" % MODE)
prev = None
for tag, m, c, lv in snaps:
    d = "-" if prev is None else str(m - prev)
    print("   step %-6s misses=%-7d (+%-6s) cached=%5.2f live=%5.2f" % (tag, m, d, c/2**30, lv/2**30))
    prev = m