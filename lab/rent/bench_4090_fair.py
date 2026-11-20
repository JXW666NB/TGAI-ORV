# -*- coding: utf-8 -*-
"""4090 公平对比：两侧各取最优设置，交替 3 轮，中位数 + 极差。

ORV: --amp off（实测最优）/ --amp gemm
PyT: --amp / 无 --amp
"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = sys.executable
ROOT = "/root/bench"
SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROUNDS = 3
STEPS = 20

ORV_ARGV = [PY, "-m", "tgai_go.train.v3_pretrain",
            "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
            "--seq", str(SEQ), "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "int8gpu", "--batch", "1", "--steps", str(STEPS),
            "--log-every", "99", "--no-final-save"]


def run(argv, env_extra=None, pat=r'"step_s_steady":\s*([0-9.]+)'):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(pat, out)
    if m is None:
        tail = " / ".join(out.strip().splitlines()[-2:])
        return None, tail[:160]
    pk = re.search(r'"pool_peak_live_gib":\s*([0-9.]+)', out)
    if pk is None:
        pk = re.search(r'"peak_gib":\s*([0-9.]+)', out)
    return float(m.group(1)), (float(pk.group(1)) if pk else None)


def pyt(amp):
    argv = [PY, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
            "--layers", "12", "--E", "4", "--vocab", "96000", "--d_ff", "3328",
            "--n_heads", "16", "--top_k", "2", "--seq", str(SEQ),
            "--steps", str(STEPS), "--ckpt", "0"]
    if amp:
        argv.append("--amp")
    return run(argv, pat=r'"step_s":\s*([0-9.]+)')


rows = {"orv_off": [], "orv_gemm": [], "pyt_amp": [], "pyt_noamp": []}

for i in range(ROUNDS):
    for key, fn in (("orv_off", lambda: run(ORV_ARGV + ["--amp", "off"],
                                            {"PYTHONPATH": "/root/patched_pkg"})),
                    ("orv_gemm", lambda: run(ORV_ARGV + ["--amp", "gemm"],
                                             {"PYTHONPATH": "/root/patched_pkg"})),
                    ("pyt_amp", lambda: pyt(True)),
                    ("pyt_noamp", lambda: pyt(False))):
        v, extra = fn()
        rows[key].append(v)
        print("  [%d] %-10s %s" % (i + 1, key, v if v is not None else "FAIL " + str(extra)),
              flush=True)

print("\n[fair] seq=%d  steps=%d  交替 %d 轮" % (SEQ, STEPS, ROUNDS))
res = {}
for k, vals in rows.items():
    ok = [v for v in vals if v is not None]
    if not ok:
        print("  %-10s 全部失败" % k)
        continue
    res[k] = {"median": round(statistics.median(ok), 4),
              "min": round(min(ok), 4), "max": round(max(ok), 4),
              "n": len(ok)}
    print("  %-10s 中位 %.4f s  (%.4f ~ %.4f, n=%d)"
          % (k, res[k]["median"], res[k]["min"], res[k]["max"], len(ok)))

if "orv_off" in res:
    orv_best = min(res[k]["median"] for k in ("orv_off", "orv_gemm") if k in res)
    pyt_best = min(res[k]["median"] for k in ("pyt_amp", "pyt_noamp") if k in res)
    print("\n  最优 ORV %.4f s  vs  最优 PyT %.4f s  =>  ORV 快 %.2f×"
          % (orv_best, pyt_best, pyt_best / orv_best))
print("\n[fair-json] " + json.dumps(res, ensure_ascii=False))
