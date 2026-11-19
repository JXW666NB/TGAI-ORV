# -*- coding: utf-8 -*-
"""对比 seq512 / seq2048 的每步主机往返与内核发射次数。"""
import json
import os
import re
import subprocess

ROOT = r"D:\orv_bench"
ORV = r"D:\orv_pip_0962\Scripts\python.exe"
STEPS = 20


def run(seq):
    cmd = [ORV, os.path.join(ROOT, "count_sync.py"),
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    r = re.search(r"^RESULT (\{.*\})$", out, re.M)
    c = re.search(r"\[counts\] (\{.*\})", out)
    if not r or not c:
        print("  seq%-5d 取数失败；输出尾部：\n%s"
              % (seq, "\n".join(out.splitlines()[-5:])))
        return None
    d = json.loads(r.group(1))
    k = json.loads(c.group(1))
    return {"step": d["step_s_steady"], "cnt": k, "mem": d["pool_peak_live_gib"]}


print("===== 主机往返 / 内核发射：每步次数（MoE 12L，20 步）=====")
rows = {}
for seq in (512, 2048):
    d = run(seq)
    rows[seq] = d
    if not d:
        continue
    k = d["cnt"]
    print("  seq%-5d 步时 %6.3f s | sync %6d 次（累计 %.3f s）| d2h %5d | h2d %5d | launch %7d"
          % (seq, d["step"], k["sync"], k["sync_s"], k["d2h"], k["h2d"],
             k["launch"]), flush=True)

a, b = rows.get(512), rows.get(2048)
if a and b:
    print()
    print("4× token 后：")
    for key in ("sync", "d2h", "h2d", "launch"):
        print("  %-7s %d → %d（%.2f×）"
              % (key, a["cnt"][key], b["cnt"][key],
                 b["cnt"][key] / max(1.0, a["cnt"][key])))
    print("  sync 累计耗时 %.3f → %.3f s（%.1f×）"
          % (a["cnt"]["sync_s"], b["cnt"]["sync_s"],
             b["cnt"]["sync_s"] / max(1e-9, a["cnt"]["sync_s"])))
