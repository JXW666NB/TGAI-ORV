# -*- coding: utf-8 -*-
"""开窗执行：停产线 → 依次跑 ORV / PyTorch 各三档 → **无论成败都拉回产线**。

每档单独起子进程：OOM 只毁那一档，不影响后续，也不会把服务留在停着的状态。

用法：`python run_window.py <bench_cfg.json> [--seqs 512,1024,2048] [--steps 10]`
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY_ORV = r"D:\orv_pip_0962\Scripts\python.exe"
PY_TORCH = r"C:\Python311\python.exe"


def sh(cmd, timeout=900):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or ""), time.time() - t0
    except subprocess.TimeoutExpired:
        return -9, "<超时 %ds 被杀>" % timeout, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("--seqs", default="512,1024,2048")
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args()
    cfgp = os.path.abspath(a.cfg)
    seqs = [int(s) for s in a.seqs.split(",") if s.strip()]

    log = []
    def note(s):
        print(s, flush=True)
        log.append(s)

    note("=== 1) 停产线 ===")
    rc, out, _ = sh([r"C:\Python311\python.exe", "-X", "utf8", r"D:\tgai_orv\stop.py"], 300)
    note("stop rc=%d %s" % (rc, (out or "").strip().splitlines()[-1:] if out else ""))
    time.sleep(5)

    results = {}
    try:
        for seq in seqs:
            for impl, py in (("orv", PY_ORV), ("torch", PY_TORCH)):
                note("\n=== %s seq=%d steps=%d ===" % (impl, seq, a.steps))
                work = os.path.join(HERE, "bench_one.py")
                rc, out, dt = sh([py, "-u", "-X", "utf8", work, cfgp,
                                  "--impl", impl, "--seq", str(seq),
                                  "--steps", str(a.steps)], 1800)
                tail = "\n".join([l for l in out.strip().splitlines() if l.strip()][-6:])
                note("rc=%d  %.0fs\n%s" % (rc, dt, tail))
                results["%s_%d" % (impl, seq)] = {"rc": rc, "wall_s": round(dt, 1),
                                                  "tail": tail}
    finally:
        note("\n=== 2) 拉回产线（无论成败）===")
        rc, out, _ = sh([r"C:\Python311\python.exe", "-X", "utf8", r"D:\tgai_orv\rs3.py"], 600)
        note("rs3 rc=%d %s" % (rc, (out or "").strip()[-200:]))

    with open(os.path.join(os.path.dirname(cfgp), "window_log.json"), "w",
              encoding="utf-8") as f:
        json.dump({"results": results, "log": log}, f, ensure_ascii=False, indent=1)
    note("\n完成，结果已写 window_log.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
