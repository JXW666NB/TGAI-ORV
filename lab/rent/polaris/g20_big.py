# -*- coding: utf-8 -*-
"""第二个 GPU 窗口：大一点的模型（~9.4M）上的 opt.step 标定 + AMP 基线 + 显存。

目的：回答"Polaris 慢 300 倍是不是只因为模型太小"。
  * 1.05M（mini）已测：Polaris opt.step 13.11 ms vs AdamW 0.04 ms
  * 本脚本：8 层 / d=256 / 9.4M，同一套计时（CUDA event，预热 3，3 次中位数）
  * 同时给 ORV 自带的 int8gpu / fp16mom 作参照（任务要求的对比项）
"""
import json
import os
import sys
import time

import numpy as np

import pbench_lib as L

L.utf8()
import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
import polaris  # noqa: F401,E402

from tgai_go.train.opt_fp16mv import Fp16MomAdamW       # noqa: E402
from tgai_go.train.int8_adamw_gpu import Int8AdamWGpu   # noqa: E402

out = {}


def peak_gib():
    try:
        return round(rt.pool_stats()["peak_live_bytes"] / 2 ** 30, 4)
    except Exception:
        return None


def build(nl, d, nh, dff, vocab=4096):
    T.manual_seed(1000)
    return T.preset.transformer(vocab=vocab, d_model=d, n_layers=nl, n_heads=nh, d_ff=dff)


def time_opt(opt, ps, X, Y, B, SEQ, reps=3, nsteps=10, warm=3, invalidate=None):
    model = None
    for _ in range(warm):
        opt.step()
    rt.sync()
    evs = []
    for _ in range(reps):
        ev0, ev1 = rt.event_create(), rt.event_create()
        rt.sync()
        rt.event_record(ev0)
        for _ in range(nsteps):
            opt.step()
            if invalidate:
                invalidate()
        rt.event_record(ev1)
        rt.event_sync(ev1)
        evs.append(rt.event_elapsed_ms(ev0, ev1) / nsteps)
        rt.event_destroy(ev0)
        rt.event_destroy(ev1)
    return evs


def time_full(model, ps, opt, X, Y, B, SEQ, reps=3, nsteps=10, warm=2):
    for _ in range(warm):
        logits = model(X[:B])
        loss = L.ce_vec(T, logits.reshape(B * SEQ, 4096), Y[:B].reshape(-1))
        for p in ps:
            p.grad = None
        loss.backward()
        opt.step()
    rt.sync()
    evs = []
    for _ in range(reps):
        ev0, ev1 = rt.event_create(), rt.event_create()
        rt.sync()
        rt.event_record(ev0)
        for _ in range(nsteps):
            logits = model(X[:B])
            loss = L.ce_vec(T, logits.reshape(B * SEQ, 4096), Y[:B].reshape(-1))
            for p in ps:
                p.grad = None
            loss.backward()
            opt.step()
        rt.event_record(ev1)
        rt.event_sync(ev1)
        evs.append(rt.event_elapsed_ms(ev0, ev1) / nsteps)
        rt.event_destroy(ev0)
        rt.event_destroy(ev1)
    return evs


CFG_BIG = dict(nl=8, d=256, nh=8, dff=1024)
CFG_MINI = dict(nl=2, d=128, nh=4, dff=512)
SEQ = 64
xs, ys = L.model_blocks(SEQ, 16)
X = np.stack(xs)
Y = np.stack(ys)
B = 2

CANDS = [
    ("AdamW", lambda ps: T.optim.AdamW(ps, lr=1e-3)),
    ("Fp16MomAdamW", lambda ps: Fp16MomAdamW(ps, lr=1e-3)),
    ("Int8AdamWGpu", lambda ps: Int8AdamWGpu(ps, lr=1e-3)),
    ("Adafactor", lambda ps: T.optim.Adafactor(ps, lr=1e-3)),
    ("Muon", lambda ps: T.optim.Muon(ps, lr=1e-2, ns_fast=True)),
    ("Lion", lambda ps: T.optim.Lion(ps, lr=1e-3)),
    ("Polaris", lambda ps: T.plugin.resolve_optimizer("Polaris")(ps, lr=1e-3)),
]

for tag, cfg in (("big_9.4M", CFG_BIG), ("mini_1.05M", CFG_MINI)):
    res = {}
    model0 = build(**cfg)
    npar = int(model0.n_params())
    del model0
    try:
        rt.pool_stats()
    except Exception:
        pass
    for name, mk in CANDS:
        try:
            model = build(**cfg)
            ps = list(model.parameters())
            # 先跑一次 fwd/bwd 让 grad 有值
            logits = model(X[:B])
            loss = L.ce_vec(T, logits.reshape(B * SEQ, 4096), Y[:B].reshape(-1))
            for p in ps:
                p.grad = None
            loss.backward()
            rt.sync()
            opt = mk(ps)
            inv = getattr(T, "fp16_shadows_invalidate", None)
            evs = time_opt(opt, ps, X, Y, B, SEQ, invalidate=inv)
            r = {"opt_step_ms_median": round(float(np.median(evs)), 4),
                 "opt_step_ms_all": [round(e, 4) for e in evs],
                 "n_params": npar, "peak_gib": peak_gib()}
            try:
                full = time_full(model, ps, mk(ps), X, Y, B, SEQ)
                r["full_step_ms_median"] = round(float(np.median(full)), 4)
                r["full_step_ms_all"] = [round(e, 4) for e in full]
                r["tokens_per_s"] = round(B * SEQ / (float(np.median(full)) / 1e3), 1)
            except Exception as e:
                r["full_err"] = "%s: %s" % (type(e).__name__, e)
            if name == "Polaris":
                r["status"] = opt.status()
            res[name] = r
            print("%-12s %-9s opt=%8.4f ms  full=%8.4f ms  tok/s=%s  n=%d"
                  % (tag, name, r["opt_step_ms_median"],
                     r.get("full_step_ms_median") or -1,
                     r.get("tokens_per_s"), npar), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            res[name] = {"error": "%s: %s" % (type(e).__name__, e)}
            print("%-12s %-9s ERR %s: %s" % (tag, name, type(e).__name__, e), flush=True)
        try:
            del model, ps, opt
        except Exception:
            pass
    out[tag] = res
    L.dump("g20_big", out)

# ---- AMP 基线：AdamW 开/关（补 g10d 里失败的那格） ----
amp_res = {}
try:
    model = build(**CFG_MINI)
    ps = list(model.parameters())
    for name, mk in (("AdamW", lambda ps: T.optim.AdamW(ps, lr=1e-3)),
                     ("Polaris", lambda ps: T.plugin.resolve_optimizer("Polaris")(ps, lr=1e-3))):
        for amp in (False, True):
            model = build(**CFG_MINI)
            ps = list(model.parameters())
            opt = mk(ps)
            curve, ts = [], []
            for i in range(120):
                t0 = time.perf_counter()
                if amp:
                    with T.amp():
                        logits = model(X[:8])
                else:
                    logits = model(X[:8])
                loss = L.ce_vec(T, logits.reshape(8 * SEQ, 4096), Y[:8].reshape(-1))
                for p in ps:
                    p.grad = None
                loss.backward()
                opt.step()
                curve.append(float(np.asarray(loss.to_numpy())))
                ts.append(time.perf_counter() - t0)
            amp_res["%s_amp_%s" % (name, "on" if amp else "off")] = {
                "last10_mean": float(np.mean(curve[-10:])),
                "min": float(min(curve)),
                "loss_at_50": curve[49], "final": curve[-1],
                "t_step_s": float(np.mean(ts[20:])),
                "all_finite": all(v == v for v in curve),
                "curve_tail": [round(v, 6) for v in curve[-20:]],
            }
            print("AMP %-16s l10=%.3e min=%.3e ms=%.2f"
                  % ("%s_amp_%s" % (name, "on" if amp else "off"),
                     amp_res["%s_amp_%s" % (name, "on" if amp else "off")]["last10_mean"],
                     amp_res["%s_amp_%s" % (name, "on" if amp else "off")]["min"],
                     amp_res["%s_amp_%s" % (name, "on" if amp else "off")]["t_step_s"] * 1e3),
                  flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    amp_res["err"] = "%s: %s" % (type(e).__name__, e)
amp_res["amp_state"] = T.amp_state() if hasattr(T, "amp_state") else None
amp_res["precision_policy"] = T.precision_policy() if hasattr(T, "precision_policy") else None
L.dump("g21_amp", amp_res)
print("DONE g20")
