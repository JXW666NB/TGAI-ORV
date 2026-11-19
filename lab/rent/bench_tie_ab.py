# -*- coding: utf-8 -*-
"""`--tie` A/B（ORV）：同一协议（预热 1 次 + 连跑 3 次取中位数）。

口径（与 `bench_stable.py` / 任务书一致）：
  12 层 MoE，d=1024 / d_ff=3328 / E=4+1 共享 / top_k=2 / n_q=16 / csa=4，
  vocab=96000、data2048_v96000、batch=1、AMP=gemm、--opt int8gpu、20 步、ckpt 关。

每个 (seq, tie) 组合：1 次预热（不计） + 3 次计时；报 中位数 / 最小 / 最大 / 极差%。
同时从 RESULT 里取 `params` / `pool_peak_live_gib` / `nvidia_peak_mib`。

用法（远端）：python bench_tie_ab.py            # 默认 seq 512 2048 都测
             python bench_tie_ab.py 512        # 只测 512
"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = r"D:\orv_pip_0962\Scripts\python.exe"
ROOT = r"D:\orv_bench"
WRAP = os.path.join(ROOT, "tie_main.py")
STEPS = 20
WARMUP = 1
RUNS = 3
OUT = os.path.join(ROOT, "res_tie_ab.json")


def one(seq, tie, layers=12):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if tie:
        env["ORV_TIE"] = "1"
    else:
        env.pop("ORV_TIE", None)
    cmd = [PY, WRAP,
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", str(layers), "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not m:
        return {"err": "no RESULT", "tail": out.splitlines()[-3:]}
    r = json.loads(m.group(1))
    mm = re.search(r'\[model\].*params=([0-9.]+)M.*tensors=(\d+)', out)
    return {"step": float(r["step_s_steady"]), "params": int(r["params"]),
            "peak_live": float(r["pool_peak_live_gib"]),
            "peak_mib": r.get("nvidia_peak_mib"),
            "tensors": int(mm.group(2)) if mm else None,
            "cfg_tie": r["cfg"].get("tie")}


def bench(seq, tie):
    tag = "seq%d_%s" % (seq, "tie" if tie else "untie")
    print("===== %s =====" % tag, flush=True)
    r = one(seq, tie)
    if "err" in r:
        print("  预热失败: %s" % (r,), flush=True)
        return None
    print("  [warmup] %.3f s  params=%.2fM  peak_live=%.2f GiB  tensors=%s cfg.tie=%r"
          % (r["step"], r["params"] / 1e6, r["peak_live"], r["tensors"], r["cfg_tie"]),
          flush=True)
    vals, last = [], r
    for i in range(RUNS):
        r = one(seq, tie)
        if "err" in r:
            print("  第 %d 次失败" % (i + 1), flush=True)
            continue
        vals.append(r["step"])
        last = r
        print("  [run %d] %.3f s  %.2f GiB" % (i + 1, r["step"], r["peak_live"]),
              flush=True)
    if not vals:
        return None
    med = statistics.median(vals)
    res = {"seq": seq, "tie": tie, "median": round(med, 3),
           "min": round(min(vals), 3), "max": round(max(vals), 3),
           "spread_pct": round(100.0 * (max(vals) - min(vals)) / med, 2),
           "runs": [round(v, 3) for v in vals],
           "params": last["params"], "tensors": last["tensors"],
           "pool_peak_live_gib": last["peak_live"],
           "nvidia_peak_mib": last["peak_mib"]}
    print("[%s] %s" % (tag, json.dumps(res, ensure_ascii=False)), flush=True)
    return res


def main():
    seqs = [int(x) for x in sys.argv[1:]] or [512, 2048]
    out = {}
    for seq in seqs:
        for tie in (False, True):
            r = bench(seq, tie)
            out["%d_%s" % (seq, "tie" if tie else "untie")] = r
    print()
    print("[tie_ab] " + json.dumps(out, ensure_ascii=False), flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[tie_ab] wrote %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
