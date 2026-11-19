# -*- coding: utf-8 -*-
"""PyTorch AdamW 的 opt.step 单独计时（与 ORV 的 opt.step 列对齐）。"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
from p30_pyt_model import MiniV3  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() and "--cpu" not in sys.argv else "cpu"
res = {"device": DEV, "torch": torch.__version__}


def bench(nl, d, nh, dff, B=2, S=64, reps=3, nsteps=20, warm=3):
    torch.manual_seed(0)
    m = MiniV3(vocab=4096, d=d, nl=nl, nh=nh, d_ff=dff).to(DEV)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    x = torch.randint(0, 4096, (B, S), device=DEV)
    y = torch.randint(0, 4096, (B * S,), device=DEV)
    loss = torch.nn.functional.cross_entropy(m(x).reshape(B * S, 4096), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    for _ in range(warm):
        opt.step()
    if DEV == "cuda":
        torch.cuda.synchronize()
    reps_ms = []
    for _ in range(reps):
        if DEV == "cuda":
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            e0.record()
            for _ in range(nsteps):
                opt.step()
            e1.record()
            e1.synchronize()
            reps_ms.append(e0.elapsed_time(e1) / nsteps)
        else:
            import time
            t0 = time.perf_counter()
            for _ in range(nsteps):
                opt.step()
            reps_ms.append((time.perf_counter() - t0) / nsteps * 1e3)
    return {"n_params": int(npar), "opt_step_ms_median": round(float(np.median(reps_ms)), 4),
            "reps_ms": [round(r, 4) for r in reps_ms]}


res["mini_1.05M"] = bench(2, 128, 4, 512)
res["big_9.4M"] = bench(8, 256, 8, 1024)
print("PYT " + json.dumps(res, ensure_ascii=True))
with open(os.path.join(OUT, "p31_pyt_optstep.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("WROTE p31")
