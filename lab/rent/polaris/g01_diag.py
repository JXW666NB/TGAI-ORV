# -*- coding: utf-8 -*-
"""GPU 诊断 1：Polaris 在 CUDA 上第一步就炸 —— 定位原因；同时给 rt 的 muon 原语清单。"""
import json
import os
import sys
import traceback

import numpy as np

import pbench_lib as L

L.utf8()
import tgai_go.train as T

out = {}
T.set_default_device("cuda")
rt = T._cuda()

# 1) ORV 内置 Muon 在 CUDA 上能不能跑（对照组）
SEQ = 64
f = np.load(r"D:\orv_bench\data2048_v4096\shard_000.npz")
ids = np.asarray(f["input_ids"]).reshape(-1)
x = ids[:SEQ].astype(np.int64)
y = ids[1:SEQ + 1].astype(np.int64)

T.manual_seed(1000)
model = T.preset.transformer(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512)
ps = list(model.parameters())


def one_step(opt, amp=False):
    if amp:
        with T.amp():
            logits = model(x)
    else:
        logits = model(x)
    loss = L.ce_vec(T, logits.reshape(SEQ, 4096), y)
    for p in ps:
        p.grad = None
    loss.backward()
    opt.step()
    rt.sync()
    return float(np.asarray(loss.to_numpy()))


# 先确认 CE 在 CUDA 上可用
try:
    lg = model(x)
    lv = L.ce_vec(T, lg.reshape(SEQ, 4096), y)
    lv.backward()
    out["ce_vec_cuda"] = "ok loss=%.4f" % float(np.asarray(lv.to_numpy()))
except Exception as e:
    out["ce_vec_cuda"] = "ERR %s: %s" % (type(e).__name__, e)
    out["ce_vec_tb"] = traceback.format_exc()[-1500:]

# 2) 内置 Muon（CUDA）
try:
    T.manual_seed(1000)
    model2 = T.preset.transformer(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512)
    ps2 = list(model2.parameters())
    opt = T.optim.Muon(ps2, lr=1e-2)
    ps = ps2
    model = model2
    out["muon_cuda_step"] = "ok loss=%.4f" % one_step(opt)
except Exception as e:
    out["muon_cuda_step"] = "ERR %s: %s" % (type(e).__name__, e)
    out["muon_tb"] = traceback.format_exc()[-1500:]

# 3) Polaris（CUDA）—— 完整 traceback
try:
    import polaris
    P = T.plugin.resolve_optimizer("Polaris")
    T.manual_seed(1000)
    model3 = T.preset.transformer(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512)
    ps3 = list(model3.parameters())
    model = model3
    ps = ps3
    opt = P(ps3, lr=1e-3)
    out["polaris_rt"] = str(opt._rt is not None)
    out["polaris_has_cuda_flag"] = bool(opt._has_cuda)
    out["polaris_step1"] = "ok loss=%.4f" % one_step(opt)
except Exception as e:
    out["polaris_step1"] = "ERR %s: %s" % (type(e).__name__, e)
    out["polaris_tb"] = traceback.format_exc()[-3000:]

# 4) rt 上 Polaris 用到的原语是否存在
names = ["muon_momentum", "sumsq_parallel", "read_scalar", "add", "scale",
         "div_scalar", "sub_scale_", "adamw", "zeros", "empty", "sync",
         "event_create", "event_record", "event_elapsed_ms", "event_sync"]
prims = {}
for n in names:
    try:
        a = getattr(rt, n)
        prims[n] = ("callable" if callable(a) else repr(a)) if a is not None else "None"
    except Exception as e:
        prims[n] = "ERR %s" % type(e).__name__
out["rt_prims"] = prims

# 5) 内置 Muon 的 CUDA step 到底调了哪个函数（看它的源码里用的名字）
try:
    import inspect
    src = inspect.getsource(T.optim.Muon._step_cuda)
    out["muon_step_cuda_src"] = src[:1800]
except Exception as e:
    out["muon_src_err"] = str(e)

L.dump("g01_diag", out)
print(json.dumps({k: v for k, v in out.items()
                  if k not in ("muon_step_cuda_src", "polaris_tb", "muon_tb",
                               "ce_vec_tb")},
                 ensure_ascii=True, indent=1, default=str)[:4000])
