# -*- coding: utf-8 -*-
"""A/B：batch==1 的 2-D 快路径（TG_AMP_BMM）对数值与速度的影响。

判据：
  1) loss 逐步差 ≤ 1e-2 绝对（fp16 输入量化会改变结果，但不该漂移）
  2) 无 NaN、末值不偏移
  3) 速度有收益
"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = r"D:\orv_pip_0962\Scripts\python.exe"
ROOT = r"D:\orv_bench"


def one(seq, mode, steps=20):
    cmd = [PY, "-m", "tgai_go.train.v3_pretrain",
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(steps), "--log-every", "99", "--no-final-save"]
    env = dict(os.environ)
    env["TG_AMP_BMM"] = mode
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'"step_s_steady":\s*([0-9.]+)', out)
    lo = re.search(r"^losses \[(.*)\]$", out, re.M)
    gn = re.findall(r"gnorm=\s*([0-9.]+)", out)
    if not m or not lo:
        return None
    return {"step": float(m.group(1)),
            "losses": [float(x) for x in lo.group(1).split(",")],
            "gnorm_first": float(gn[0]) if gn else None,
            "gnorm_last": float(gn[-1]) if gn else None}


report = {}
for seq in (512, 2048):
    a = one(seq, "0")
    b = one(seq, "1")
    if not a or not b:
        print("seq%d 取数失败" % seq)
        continue
    n = min(len(a["losses"]), len(b["losses"]))
    diffs = [abs(a["losses"][i] - b["losses"][i]) for i in range(n)]
    report[seq] = {
        "step_off": round(a["step"], 3), "step_on": round(b["step"], 3),
        "delta_ms": round((a["step"] - b["step"]) * 1000, 1),
        "loss_first_off": a["losses"][0], "loss_first_on": b["losses"][0],
        "loss_last_off": a["losses"][-1], "loss_last_on": b["losses"][-1],
        "max_abs_loss_diff": round(max(diffs), 6),
        "nan": any(x != x for x in b["losses"]),
        "gnorm_off": a["gnorm_last"], "gnorm_on": b["gnorm_last"],
    }
    print("seq%d: %.3f → %.3f s（%+.1f ms） max|Δloss|=%.6f  gnorm %.4f→%.4f"
          % (seq, a["step"], b["step"], -report[seq]["delta_ms"] * -1,
             report[seq]["max_abs_loss_diff"],
             a["gnorm_last"] or 0, b["gnorm_last"] or 0), flush=True)

print()
print("[ab_amp_bmm] " + json.dumps(report, ensure_ascii=False))
