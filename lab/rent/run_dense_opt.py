# -*- coding: utf-8 -*-
"""PyTorch 侧：优化器单独计时 + 稠密模型长序列扫描。

两件事：
 1) `opt_only_pyt.py` 的等价物：只测 AdamW+裁剪 一步（O(参数量) 的固定开销）
 2) 稠密模型 seq 扫描（MoE 之外的对照组）
"""
import json
import os
import re
import subprocess
import sys

ROOT = r"D:\orv_bench"
PYT = r"C:\Python311\python.exe"
STEPS = 3

OPT_ONLY = r'''
import json, time, numpy as np, torch, sys
sys.path.insert(0, r"D:\orv_bench")
import model_v3 as M
LAYERS = int(sys.argv[1]); DENSE = (len(sys.argv) > 2 and sys.argv[2] == "dense")
d_ff = 13312 if DENSE else 3328
E = 1 if DENSE else 4
cfg = M.TGAIV3Config(vocab_size=96000, d_model=1024, n_layers=LAYERS, n_heads=16,
                     n_kv_heads=2, d_ff=d_ff, n_experts=E, n_activated=2 if not DENSE else 1,
                     n_shared_experts=1, max_seq_len=512, moe_balance="sign")
m = M.TGAIV3Model(cfg).cuda()
ps = list(m.parameters())
n = sum(p.numel() for p in ps)
for p in ps:
    p.grad = torch.full_like(p, 1e-3)
opt = torch.optim.AdamW(ps, lr=3e-4)
torch.cuda.synchronize()
ts = []
for i in range(4):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(ps, 1.0)
    opt.step()
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    if i >= 1: ts.append(dt)
ts.sort()
print(json.dumps({"impl": "pytorch", "layers": LAYERS, "dense": DENSE,
                  "params": n, "opt_step_s": ts[len(ts)//2]}))
'''


def run(cmd, stdin=None):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", input=stdin, cwd=ROOT)
    return (p.stdout or "") + (p.stderr or "")


def opt_only_pyt(layers, dense):
    out = run([PYT, "-c", OPT_ONLY, str(layers), "dense" if dense else "moe"])
    m = re.search(r"\{.*\}", out)
    return json.loads(m.group(0)) if m else None


def dense_step(seq, layers=12):
    cmd = [PYT, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
           "--layers", str(layers), "--E", "1", "--vocab", "96000",
           "--d_ff", "13312", "--n_heads", "16", "--top_k", "1",
           "--seq", str(seq), "--steps", str(STEPS)]
    out = run(cmd)
    m = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not m:
        return None
    d = json.loads(m.group(1))
    return {"step": d["step_s"], "mem": d["peak_mib"] / 1024.0}


def orv_dense(seq, layers=12):
    cmd = [r"D:\orv_pip_0962\Scripts\python.exe", os.path.join(ROOT, "split_orv.py"),
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", str(layers), "--E", "1",
           "--n_q", "16", "--d_ff", "13312", "--top_k", "1", "--csa", "4",
           "--opt", "int8gpu", "--steps", str(STEPS), "--log-every", "99",
           "--no-final-save", "--no-shared"]
    out = run(cmd)
    r = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not r:
        return None
    d = json.loads(r.group(1))
    return {"step": d["step_s_steady"], "mem": d["pool_peak_live_gib"]}


print("===== 稠密模型 seq 扫描（12 层，d_ff=13312）=====")
for seq in (512, 1024, 2048):
    a = orv_dense(seq)
    b = dense_step(seq)
    if a and b:
        print("seq%-5d ORV %6.3f s (%8.1f tok/s, %5.1f GiB)   PyT %6.3f s (%8.1f tok/s, %5.1f GiB)   ORV %4.2fx"
              % (seq, a["step"], seq / a["step"], a["mem"],
                 b["step"], seq / b["step"], b["mem"], b["step"] / a["step"]), flush=True)
    else:
        print("seq%-5d ORV=%s PyT=%s" % (seq, a, b), flush=True)

print()
print("===== 优化器单独一步（O(参数量) 的固定开销）=====")
for layers, dense in ((12, False), (12, True)):
    out = run([r"D:\orv_pip_0962\Scripts\python.exe",
               os.path.join(ROOT, "opt_only_orv.py"), str(layers),
               "dense" if dense else "moe"])
    m = re.search(r"\{.*\}", out)
    a = json.loads(m.group(0)) if m else None
    b = opt_only_pyt(layers, dense)
    tag = "稠密12L" if dense else "MoE12L"
    if a and b:
        print("%s  参数 %.0fM | ORV 优化器 %6.4f s | PyT 优化器 %6.4f s | PyT 是 ORV 的 %.1f×"
              % (tag, a["params"] / 1e6, a["opt_step_s"], b["opt_step_s"],
                 b["opt_step_s"] / a["opt_step_s"]), flush=True)
    else:
        print("%s  ORV=%s PyT=%s" % (tag, a, b), flush=True)
