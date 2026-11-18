# -*- coding: utf-8 -*-
"""激活精度 A/B：打开 bf16 激活存储（TG_BF16_ACT=1）看 seq2048 的步时变化。

若显著变快 ⇒ 长序列的瓶颈是**激活读写带宽**（不是注意力、也不是 head/loss）。
注意：bf16 存储会改变数值，这里只当性能探针，正确性另行验证。
"""
import json
import os
import re
import subprocess

ROOT = r"D:\orv_bench"
PY = r"D:\orv_pip_0962\Scripts\python.exe"
STEPS = 3


def one(seq, bf16):
    cmd = [PY, os.path.join(ROOT, "split_orv.py"),
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--steps", str(STEPS), "--log-every", "99",
           "--no-final-save"]
    env = dict(os.environ)
    if bf16:
        env["TG_BF16_ACT"] = "1"
    else:
        env.pop("TG_BF16_ACT", None)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    r = re.search(r"^RESULT (\{.*\})$", out, re.M)
    m = re.search(r"\[split\] (\{.*\})", out)
    if not r:
        return None
    res = json.loads(r.group(1))
    attn = json.loads(m.group(1))["attn_s_total"] / float(STEPS) if m else float("nan")
    return {"step": res["step_s_steady"], "attn": attn,
            "rest": res["step_s_steady"] - attn, "mem": res["pool_peak_live_gib"]}


for seq in (512, 2048):
    a = one(seq, False)
    b = one(seq, True)
    for tag, d in (("fp32存储", a), ("bf16存储", b)):
        if d is None:
            print("seq%-5d %-9s FAIL" % (seq, tag))
        else:
            print("seq%-5d %-9s step=%6.3f s  attn=%6.3f  rest=%6.3f  mem=%5.1f GiB"
                  % (seq, tag, d["step"], d["attn"], d["rest"], d["mem"]), flush=True)
    if a and b:
        print("  ⇒ bf16 激活：步时 %.3f→%.3f s（%+.1f%%），内存 %.1f→%.1f GiB（%+.1f%%）"
              % (a["step"], b["step"], 100 * (b["step"] - a["step"]) / a["step"],
                 a["mem"], b["mem"], 100 * (b["mem"] - a["mem"]) / a["mem"]), flush=True)
