# -*- coding: utf-8 -*-
"""给 Polaris 做 lr 扫描，与 AdamW 的最优 lr 做公平对比。

为什么必须扫描：Muon 系优化器的更新标度与 AdamW 不同（它按谱范数归一），
用同一个 lr 比是无效对比 —— 上一轮同 lr=1e-2 下 Polaris 差 19 倍就是这么来的。
"""
import json
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
T.plugin.autoload()
import polaris                                     # noqa: F401,E402

Polaris = T.plugin.resolve_optimizer("Polaris")
D_IN, D_OUT, N = 128, 256, 8
STEPS = 80

rng = np.random.default_rng(3)
x_np = rng.normal(size=(D_IN, N)).astype(np.float32)
t_np = rng.normal(size=(D_OUT, N)).astype(np.float32)


def run(make_opt, lr, steps=STEPS):
    rng2 = np.random.default_rng(7)                # 每组相同初始化
    w = T.tensor(rng2.normal(0, 0.02, (D_OUT, D_IN)).astype(np.float32),
                 requires_grad=True)
    x = T.tensor(x_np)
    t = T.tensor(t_np)
    opt = make_opt([w], lr)
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        d = T.matmul(w, x) - t
        loss = (d * d).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(rt.read_scalar(loss._gpu)))
    rt.sync()
    dt = time.perf_counter() - t0
    return losses[0], sum(losses[-10:]) / 10.0, dt


print("===== AdamW lr 扫描 =====")
best_a = None
for lr in (3e-2, 1e-2, 3e-3, 1e-3):
    try:
        f, tail, dt = run(lambda ps, l: T.optim.AdamW(ps, lr=l), lr)
        ok = np.isfinite(tail)
        print("  lr=%.0e  末段 %.6f  %.3f s" % (lr, tail, dt), flush=True)
        if ok and (best_a is None or tail < best_a[1]):
            best_a = (lr, tail, dt)
    except Exception as e:                          # noqa: BLE001
        print("  lr=%.0e  FAIL %s" % (lr, e), flush=True)

print("===== Polaris lr 扫描 =====")
best_p = None
for lr in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4):
    try:
        f, tail, dt = run(lambda ps, l: Polaris(ps, lr=l), lr)
        ok = np.isfinite(tail)
        print("  lr=%.0e  末段 %.6f  %.3f s" % (lr, tail, dt), flush=True)
        if ok and (best_p is None or tail < best_p[1]):
            best_p = (lr, tail, dt)
    except Exception as e:                          # noqa: BLE001
        print("  lr=%.0e  FAIL %s" % (lr, e), flush=True)

print()
print("[best] " + json.dumps({
    "adamw": {"lr": best_a[0], "tail": best_a[1], "s": best_a[2]} if best_a else None,
    "polaris": {"lr": best_p[0], "tail": best_p[1], "s": best_p[2]} if best_p else None,
}, default=str))
if best_a and best_p:
    print("各自最优下：Polaris 末段是 AdamW 的 %.2f 倍（<1 更好）"
          % (best_p[1] / max(1e-12, best_a[1])))
