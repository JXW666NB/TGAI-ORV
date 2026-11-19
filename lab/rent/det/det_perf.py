# -*- coding: utf-8 -*-
"""端到端性能：直接复用官方 `bench_stable.py`（预热 1 次 + 3 次取中位数），
只多一层"开/关 TG_DETERMINISTIC"的外壳。

为什么用 CUDA event 的旁证：`bench_stable` 统计的是每步的 GPU 侧稳态步时
（`v3_pretrain` 每步末尾 `rt.sync()` 后取差）；更细的算子级对比见
`det_perf_kernel.py`（那条**完全用 rt.event_* 打点**，不做前后 sync）。

跑法：D:\\orv_bench\\remote_py.bat D:\\orv_bench\\det_perf.py
"""
import json
import os
import sys

sys.path.insert(0, r"D:\orv_bench")

import bench_stable as B                                     # noqa: E402

OUT = r"D:\orv_bench\det_perf.json"


def main():
    seqs = [512, 2048]
    out = {}
    for mode in ("off", "on"):
        if mode == "on":
            os.environ["TG_DETERMINISTIC"] = "1"
        else:
            os.environ.pop("TG_DETERMINISTIC", None)
        for seq in seqs:
            key = "%s_%d" % (mode, seq)
            print("===== %s (TG_DETERMINISTIC=%s) ====="
                  % (key, os.environ.get("TG_DETERMINISTIC", "<unset>")), flush=True)
            out[key] = B.bench(seq)
            print("  -> %s" % (out[key],), flush=True)
    print("\n[det_perf]")
    for seq in seqs:
        off = out.get("off_%d" % seq) or {}
        on = out.get("on_%d" % seq) or {}
        if not off or not on:
            print("  seq%d: 缺数据" % seq)
            continue
        d = on["median"] - off["median"]
        print("  seq%-5d off %.3f s (%.3f..%.3f, 极差 %.1f%%)   on %.3f s "
              "(%.3f..%.3f, %.1f%%)   Δ %+.3f s (%+.1f%%)"
              % (seq, off["median"], off["min"], off["max"], off["spread_pct"],
                 on["median"], on["min"], on["max"], on["spread_pct"],
                 d, 100.0 * d / off["median"]))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[det_perf] wrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
