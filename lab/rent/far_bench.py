# -*- coding: utf-8 -*-
"""总基准：ORV（tie 关/开）× {512,2048} + PyTorch × {512,2048}，同一协议。

协议（任务书）：预热 1 次（不计） + 连跑 3 次取中位数；20 步；ckpt 关；batch=1；AMP 开。

口径明细（报告里必须写）：
  ORV      : d=1024 d_ff=3328 layers=12 E=4(+1 共享) top_k=2 n_q=16 csa=4
             vocab=96000 data=data2048_v96000 --opt int8gpu --amp gemm
             RESULT 的 `step_s_steady` = secs[2:] 的均值（step 2..19）
  PyTorch  : 同 d/layers/E/d_ff/n_heads=16/top_k=2/vocab=96000 --ckpt 0 --amp
             `TGAIV3Config.tie_embeddings` **默认 True**（= ORV 的 --tie）
             `step_s` = 全部 20 步的均值（含第 0 步）⇒ 额外算 step2..19 的均值对照

产出：D:\\orv_bench\\res_final_ab.json
"""
import json
import os
import re
import statistics
import subprocess
import sys

ORV = r"D:\orv_pip_0962\Scripts\python.exe"
PYT = r"C:\Python311\python.exe"
ROOT = r"D:\orv_bench"
WRAP = os.path.join(ROOT, "tie_main.py")
OUT = os.path.join(ROOT, "res_final_ab.json")
STEPS = 20
RUNS = 3
LAYERS = 12


def _orv_one(seq, tie):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if tie:
        env["ORV_TIE"] = "1"
    else:
        env.pop("ORV_TIE", None)
    cmd = [ORV, WRAP,
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", str(LAYERS), "--E", "4",
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
    return {"step": float(r["step_s_steady"]), "params": int(r["params"]),
            "peak_live": float(r["pool_peak_live_gib"]),
            "peak_mib": r.get("nvidia_peak_mib"),
            "cfg_tie": bool(r["cfg"].get("tie", False)),
            "cfg_vocab": r["cfg"].get("vocab")}


def _pyt_one(seq):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [PYT, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
           "--layers", str(LAYERS), "--E", "4", "--vocab", "96000",
           "--d_ff", "3328", "--n_heads", "16", "--top_k", "2",
           "--seq", str(seq), "--steps", str(STEPS), "--ckpt", "0", "--amp"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not m:
        return {"err": "no RESULT", "tail": out.splitlines()[-3:]}
    r = json.loads(m.group(1))
    per = [float(x) for x in re.findall(r"^step \d+ loss=[-\d.]+ ([\d.]+)s", out, re.M)]
    steady = float(statistics.mean(per[2:])) if len(per) > 2 else None
    return {"step": float(r["step_s"]), "step_steady": steady,
            "params": int(r["n_params"]), "peak_mib": r.get("peak_mib"),
            "torch_peak_mib": r.get("torch_peak_mib")}


def bench(kind, seq, tie=False):
    tag = "%s_seq%d_%s" % (kind, seq, "tie" if tie else "untie")
    fn = (lambda: _pyt_one(seq)) if kind == "pyt" else (lambda: _orv_one(seq, tie))
    print("===== %s =====" % tag, flush=True)
    w = fn()
    if "err" in w:
        print("  预热失败: %s" % (w,), flush=True)
        return None
    print("  [warmup] step=%.3f params=%.2fM" % (w["step"], w["params"] / 1e6), flush=True)
    vals, last = [], w
    for i in range(RUNS):
        r = fn()
        if "err" in r:
            print("  第 %d 次失败" % (i + 1), flush=True)
            continue
        vals.append(r["step"])
        last = r
        print("  [run %d] %.3f s" % (i + 1, r["step"]), flush=True)
    if not vals:
        return None
    med = statistics.median(vals)
    res = {"kind": kind, "seq": seq, "tie": tie, "median": round(med, 3),
           "min": round(min(vals), 3), "max": round(max(vals), 3),
           "spread_pct": round(100.0 * (max(vals) - min(vals)) / med, 2),
           "runs": [round(v, 3) for v in vals],
           "params": last["params"], "peak_mib": last.get("peak_mib")}
    if kind == "orv":
        res["pool_peak_live_gib"] = last["peak_live"]
        res["cfg_tie"] = last["cfg_tie"]
    else:
        res["step_steady"] = round(last["step_steady"], 3) if last.get("step_steady") else None
        res["torch_peak_mib"] = last.get("torch_peak_mib")
    print("[%s] %s" % (tag, json.dumps(res, ensure_ascii=False)), flush=True)
    return res


def main():
    out = {}
    for seq in (512, 2048):
        out["orv_%d_untie" % seq] = bench("orv", seq, False)
        out["orv_%d_tie" % seq] = bench("orv", seq, True)
        out["pyt_%d_tie" % seq] = bench("pyt", seq, False)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    print()
    print("[final] " + json.dumps(out, ensure_ascii=False), flush=True)
    print("[final] wrote %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
