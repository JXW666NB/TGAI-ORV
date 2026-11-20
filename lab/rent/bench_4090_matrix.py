# -*- coding: utf-8 -*-
"""4090 宣传用基准矩阵：多序列 × ORV/PyT，各取最优设置，交替 3 轮。

输出 README 可直接用的 JSON：中位/极差、tok/s、ORV 峰值显存、是否 OOM。
"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = sys.executable
ROOT = "/root/bench"
PKG = "/root/patched_pkg"
STEPS = 20
ROUNDS = 3

CELLS = [(512, False), (1024, False), (1024, True), (2048, True)]


def run(argv, env_extra):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(env_extra)
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'"step_s_steady":\s*([0-9.]+)', out) or \
        re.search(r'"step_s":\s*([0-9.]+)', out)
    if m is None:
        low = out.lower()
        why = "OOM" if ("out of memory" in low or "显存不足" in out) else "FAIL"
        return None, None, why
    pk = re.search(r'"pool_peak_live_gib":\s*([0-9.]+)', out)
    return float(m.group(1)), (float(pk.group(1)) if pk else None), "ok"


def orv(seq, ckpt):
    argv = [PY, "-m", "tgai_go.train.v3_pretrain",
            "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
            "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "int8gpu", "--amp", "off", "--batch", "1",
            "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    e = {"PYTHONPATH": PKG}
    if ckpt:
        e["TG_GRAD_CKPT"] = "1"
    return run(argv, e)


def pyt(seq, ckpt):
    argv = [PY, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
            "--layers", "12", "--E", "4", "--vocab", "96000", "--d_ff", "3328",
            "--n_heads", "16", "--top_k", "2", "--seq", str(seq),
            "--steps", str(STEPS), "--ckpt", "1" if ckpt else "0"]
    return run(argv, {})


print("=== 4090 基准矩阵（%d 步，交替 %d 轮，两侧各取最优）===" % (STEPS, ROUNDS))
table = {}
for seq, ckpt in CELLS:
    key = "seq%d%s" % (seq, "+ckpt" if ckpt else "")
    o, p = [], []
    peak = None
    ok_o, ok_p = "ok", "ok"
    for i in range(ROUNDS):
        a, pa, sa = orv(seq, ckpt)
        b, _pb, sb = pyt(seq, ckpt)
        ok_o, ok_p = sa, sb
        if a:
            o.append(a)
            peak = pa or peak
        if b:
            p.append(b)
        print("  [%d] %-14s ORV %-9s | PyT %-9s"
              % (i + 1, key, ("%.4f" % a) if a else sa,
                 ("%.4f" % b) if b else sb), flush=True)
    cell = {"ckpt": ckpt}
    if o:
        cell.update({"orv": round(statistics.median(o), 4),
                     "orv_range": [round(min(o), 4), round(max(o), 4)],
                     "orv_tok_s": int(round(seq / statistics.median(o))),
                     "orv_peak_gib": peak})
    else:
        cell["orv"] = ok_o
    if p:
        cell.update({"pyt": round(statistics.median(p), 4),
                     "pyt_range": [round(min(p), 4), round(max(p), 4)],
                     "pyt_tok_s": int(round(seq / statistics.median(p)))})
    else:
        cell["pyt"] = ok_p
    if o and p:
        cell["speedup"] = round(statistics.median(p) / statistics.median(o), 2)
        print("  => %-14s ORV %.4f  PyT %.4f  ORV 快 %.2f×"
              % (key, cell["orv"], cell["pyt"], cell["speedup"]))
    table[key] = cell

print("\n[matrix] " + json.dumps(table, ensure_ascii=False))
