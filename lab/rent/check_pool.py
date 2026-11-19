# -*- coding: utf-8 -*-
"""确认自适应池上限生效 + 跑一次 seq2048 基准（不带任何探针）。"""
import json
import os
import re
import subprocess

os.environ.pop("TG_POOL_GB", None)

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
cap = rt._pool_cap / (1024 ** 3)
try:
    total = rt._device_total_bytes() / (1024 ** 3)
except Exception as e:                                   # noqa: BLE001
    total = "读不到: %s" % e
print(json.dumps({"pool_cap_gib": round(cap, 2), "device_total_gib": total}))

cmd = [r"D:\orv_pip_0962\Scripts\python.exe", "-m", "tgai_go.train.v3_pretrain",
       "--data", r"D:\orv_bench\data2048_v96000", "--vocab", "96000",
       "--seq", "2048", "--d", "1024", "--layers", "12", "--E", "4",
       "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
       "--opt", "int8gpu", "--amp", "gemm", "--batch", "1", "--steps", "20",
       "--log-every", "99", "--no-final-save"]
env = dict(os.environ)
env.pop("TG_POOL_GB", None)
p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                   errors="replace", env=env, cwd=r"D:\orv_bench")
out = (p.stdout or "") + (p.stderr or "")
m = re.search(r'"step_s_steady":\s*([0-9.]+)', out)
pk = re.search(r'"pool_peak_live_gib":\s*([0-9.]+)', out)
print(json.dumps({"step_s_steady": m.group(1) if m else None,
                  "peak_live_gib": pk.group(1) if pk else None}))
if not m:
    print("\n".join(out.splitlines()[-6:]))
