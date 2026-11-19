# -*- coding: utf-8 -*-
"""GPU 主脚本：A) 玩具矩阵  B) 迷你 V3 过拟合矩阵  C) 速度/显存  D) AMP 开/关。

输出分阶段写盘：A -> g10a_toy.json, B -> g10b_model.json, C -> g10c_speed.json,
D -> g10d_amp.json。这样窗口到点也能拿到已完成阶段的结果。
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
import polaris  # noqa: F401,E402  —— 注册 T.optim.Polaris

P = T.plugin.resolve_optimizer("Polaris")
assert P is not None
GPU_ORDS = ["AdamW", "Adafactor", "Muon", "Lion", "Polaris"]   # LAMB: CPU only

STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"


def fmt(v, p=6):
    return ("%." + str(p) + "g") % v if isinstance(v, (int, float)) else str(v)


def peak_gib():
    try:
        return round(rt.pool_stats()["peak_live_bytes"] / 2 ** 30, 4)
    except Exception:
        return None


# =====================================================================  A: toy
if STAGE in ("all", "a"):
    STEPS = 600
    recs = []
    t_all = time.perf_counter()
    for name in GPU_ORDS:
        for lr in L.GRID[name]:
            kw = {"ns_fast": True} if name == "Muon" else {}
            try:
                r = L.run_toy(T, name, lr, seed=0, steps=STEPS, opt_kw=kw)
                recs.append(r)
                print("A %-9s lr=%-7.0e l10=%-11s excess=%-11s st<=1e-3=%-5s fin=%s"
                      % (name, lr, fmt(r["last10_mean"], 7), fmt(r["excess_last10"], 4),
                         fmt(r["steps_to_excess_1e-3"], 5), r["all_finite"]), flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                recs.append({"task": "toy", "opt": name, "lr": lr,
                             "error": "%s: %s" % (type(e).__name__, e)})
    L.dump("g10a_toy", {"records": recs, "steps": STEPS, "device": "cuda",
                        "loss_star": L.toy_problem()[5],
                        "wall_s": round(time.perf_counter() - t_all, 1)})

# ===========================================================  B: mini 过拟合
SEQ, NB, STEPS = 64, 8, 200
xs, ys = L.model_blocks(SEQ, NB)
X = np.stack(xs)
Y = np.stack(ys)
BATCH = NB
GRID_B = {"AdamW": [1e-3, 3e-3, 1e-2, 3e-2],
          "Adafactor": [3e-3, 1e-2, 3e-2, 1e-1],
          "Muon": [1e-3, 3e-3, 1e-2, 3e-2],
          "Lion": [3e-4, 1e-3, 3e-3, 1e-2],
          "Polaris": [3e-4, 1e-3, 3e-3, 1e-2]}


def train_overfit(name, lr, steps=STEPS, amp=False, seed=0, collect=True):
    T.manual_seed(1000 + seed)
    model = T.preset.transformer(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512)
    ps = list(model.parameters())
    kw = {"ns_fast": True} if name == "Muon" else {}
    opt = L.make_opt(T, name, ps, lr, **kw)
    curve, ts = [], []
    for i in range(steps):
        t0 = time.perf_counter()
        if amp:
            with T.amp():
                logits = model(X[:BATCH])
        else:
            logits = model(X[:BATCH])
        loss = L.ce_vec(T, logits.reshape(BATCH * SEQ, 4096), Y[:BATCH].reshape(-1))
        for p in ps:
            p.grad = None
        loss.backward()
        opt.step()
        curve.append(float(np.asarray(loss.to_numpy())))
        ts.append(time.perf_counter() - t0)
    r = L.summarize(curve, t_steps=ts)
    r.update({"task": "mini_v3_overfit", "opt": name, "lr": lr, "seed": seed,
              "steps": steps, "batch": BATCH, "seq": SEQ, "amp": amp,
              "tokens_per_step": BATCH * SEQ,
              "tokens_per_s": (BATCH * SEQ / r["t_step_s"]) if r.get("t_step_s") else None,
              "curve": [round(v, 6) if v == v else None for v in curve] if collect else None})
    if name == "Polaris" and hasattr(opt, "status"):
        r["opt_status"] = opt.status()
    return r


def label(x):
    return (x or "").upper()


if STAGE in ("all", "b"):
    recs = []
    t_all = time.perf_counter()
    for name in GPU_ORDS:
        for lr in GRID_B[name]:
            try:
                r = train_overfit(name, lr)
                recs.append(r)
                print("B %-9s lr=%-7.0e l10=%-10s l25=%-10s l50=%-10s min=%-10s "
                      "ms/step=%-6.1f fin=%s%s"
                      % (name, lr, fmt(r["last10_mean"], 6), fmt(r.get("loss_at_25"), 6),
                         fmt(r.get("loss_at_50"), 6), fmt(r["min"], 6),
                         (r["t_step_s"] or 0) * 1e3, r["all_finite"],
                         (" nssaved=%s" % r["opt_status"]["ns_saved_pct"])
                         if r.get("opt_status") else ""), flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                recs.append({"task": "mini_v3_overfit", "opt": name, "lr": lr,
                             "error": "%s: %s" % (type(e).__name__, e)})
    L.dump("g10b_model", {"records": recs, "steps": STEPS, "device": "cuda",
                          "batch": BATCH, "seq": SEQ,
                          "wall_s": round(time.perf_counter() - t_all, 1)})

# ==============================================================  C: 速度/显存
if STAGE in ("all", "c"):
    REPS, NSTEPS, WARM = 3, 20, 5
    sp = {}
    for name in ["AdamW", "Adafactor", "Muon", "Lion", "Polaris"]:
        kw = {"ns_fast": True} if name == "Muon" else {}
        try:
            T.manual_seed(1000)
            model = T.preset.transformer(vocab=4096, d_model=128, n_layers=2,
                                         n_heads=4, d_ff=512)
            ps = list(model.parameters())
            opt = L.make_opt(T, name, ps, 1e-3, **kw)
            # 预热
            for _ in range(WARM):
                logits = model(X[:BATCH])
                loss = L.ce_vec(T, logits.reshape(BATCH * SEQ, 4096), Y[:BATCH].reshape(-1))
                for p in ps:
                    p.grad = None
                loss.backward()
                opt.step()
            rt.sync()
            base_peak = peak_gib()
            times, times_opt = [], []
            for rep in range(REPS):
                ev0, ev1 = rt.event_create(), rt.event_create()
                rt.sync()
                rt.event_record(ev0)
                for _ in range(NSTEPS):
                    logits = model(X[:BATCH])
                    loss = L.ce_vec(T, logits.reshape(BATCH * SEQ, 4096),
                                    Y[:BATCH].reshape(-1))
                    for p in ps:
                        p.grad = None
                    loss.backward()
                    opt.step()
                rt.event_record(ev1)
                rt.event_sync(ev1)
                times.append(rt.event_elapsed_ms(ev0, ev1) / NSTEPS)
                rt.event_destroy(ev0)
                rt.event_destroy(ev1)
            # 单独量 opt.step（梯度固定不动）
            logits = model(X[:BATCH])
            loss = L.ce_vec(T, logits.reshape(BATCH * SEQ, 4096), Y[:BATCH].reshape(-1))
            for p in ps:
                p.grad = None
            loss.backward()
            for rep in range(REPS):
                ev0, ev1 = rt.event_create(), rt.event_create()
                rt.sync()
                rt.event_record(ev0)
                for _ in range(NSTEPS):
                    opt.step()
                    T.fp16_shadows_invalidate()
                rt.event_record(ev1)
                rt.event_sync(ev1)
                times_opt.append(rt.event_elapsed_ms(ev0, ev1) / NSTEPS)
                rt.event_destroy(ev0)
                rt.event_destroy(ev1)
            sp[name] = {
                "step_ms_median": round(float(np.median(times)), 3),
                "step_ms_all": [round(t, 3) for t in times],
                "opt_step_ms_median": round(float(np.median(times_opt)), 3),
                "opt_step_ms_all": [round(t, 3) for t in times_opt],
                "tokens_per_s": round(BATCH * SEQ / (float(np.median(times)) / 1e3), 1),
                "peak_gib": peak_gib(), "peak_gib_before": base_peak,
            }
            print("C %-9s step=%.2fms opt=%.2fms tok/s=%.0f peak=%.4fGiB"
                  % (name, sp[name]["step_ms_median"], sp[name]["opt_step_ms_median"],
                     sp[name]["tokens_per_s"], sp[name]["peak_gib"] or 0), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            sp[name] = {"error": "%s: %s" % (type(e).__name__, e)}
    L.dump("g10c_speed", {"speed": sp, "batch": BATCH, "seq": SEQ, "reps": REPS,
                          "nsteps": NSTEPS, "warm": WARM,
                          "timer": "cuda events (event_record/event_elapsed_ms)"})

# ================================================================  D: AMP 开/关
if STAGE in ("all", "d"):
    res = {}
    for name in ["AdamW", "Polaris"]:
        for amp in (False, True):
            key = "%s_amp_%s" % (name, "on" if amp else "off")
            try:
                r = train_overfit(name, 1e-3, steps=120, amp=amp)
                res[key] = {k: r[k] for k in
                            ("last10_mean", "min", "loss_at_50", "loss_at_100",
                             "t_step_s", "all_finite", "nan_step", "opt_status")}
                print("D %-16s l10=%-10s l50=%-10s ms/step=%-6.1f fin=%s"
                      % (key, fmt(r["last10_mean"], 6), fmt(r.get("loss_at_50"), 6),
                         (r["t_step_s"] or 0) * 1e3, r["all_finite"]), flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                res[key] = {"error": "%s: %s" % (type(e).__name__, e)}
    try:
        res["amp_state"] = T.amp_state() if hasattr(T, "amp_state") else None
        res["precision_policy"] = T.precision_policy() if hasattr(T, "precision_policy") else None
    except Exception as e:
        res["amp_meta_err"] = str(e)
    L.dump("g10d_amp", res)

print("DONE stage=%s" % STAGE)
