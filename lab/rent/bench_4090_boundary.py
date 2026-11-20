# -*- coding: utf-8 -*-
"""4090 内存边界：seq 逐级放大，两侧同开梯度检查点，看谁能跑。

ORV: TG_GRAD_CKPT=1 --amp off（最优）
PyT: --ckpt 1，无 --amp（最优）
"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = sys.executable
ROOT = "/root/bench"
STEPS = 10


def run(argv, env_extra=None, pat=r'"step_s_steady":\s*([0-9.]+)'):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    try:
        p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=ROOT, env=env, timeout=3600)
    except subprocess.TimeoutExpired:
        return None, "timeout", None
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(pat, out)
    if m is None:
        low = out.lower()
        why = "OOM" if ("out of memory" in low or "显存不足" in out) else "fail"
        return None, why, out.strip().splitlines()[-1][:110] if out.strip() else ""
    pk = re.search(r'"pool_peak_live_gib":\s*([0-9.]+)', out) or \
        re.search(r'"peak_gib":\s*([0-9.]+)', out)
    return float(m.group(1)), "ok", (float(pk.group(1)) if pk else None)


def orv(seq):
    argv = [PY, "-m", "tgai_go.train.v3_pretrain",
            "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
            "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "int8gpu", "--amp", "off", "--batch", "1",
            "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    return run(argv, {"PYTHONPATH": "/root/patched_pkg", "TG_GRAD_CKPT": "1"})


def pyt(seq):
    argv = [PY, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
            "--layers", "12", "--E", "4", "--vocab", "96000", "--d_ff", "3328",
            "--n_heads", "16", "--top_k", "2", "--seq", str(seq),
            "--steps", str(STEPS), "--ckpt", "1"]
    return run(argv, pat=r'"step_s":\s*([0-9.]+)')


print("=== 4090 内存边界（两侧同开梯度检查点，%d 步）===" % STEPS)


def fmt(r):
    if r[1] != "ok":
        return r[1]
    return "%.3fs%s" % (r[0], ("/%.1fGiB" % r[2]) if r[2] is not None else "")


print("  %-6s %-28s %-28s" % ("seq", "ORV (int8gpu+ckpt)", "PyT (noamp+ckpt)"))
res = {}
for seq in [4096, 2048, 1024]:
    a = orv(seq)
    b = pyt(seq)
    res[seq] = {"orv": a, "pyt": b}
    print("  %-6d %-28s %-28s" % (seq, fmt(a), fmt(b)), flush=True)

print("\n[boundary] " + json.dumps(
    {str(k): {"orv": v["orv"][:2] + (v["orv"][2],),
              "pyt": v["pyt"][:2] + (v["pyt"][2],)} for k, v in res.items()},
    ensure_ascii=False, default=str))
