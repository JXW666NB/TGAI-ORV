# -*- coding: utf-8 -*-
"""验证社区插件 Polaris：注册成功？能在 CUDA 上训练？不发散？

任务：小回归 y = W·x，目标 t 固定，最小化 ‖W·x − t‖²。
对照：同样的初始化与数据，AdamW vs Polaris（各 50 步），比末段 loss 与步时。
"""
import json
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()

report = {}

# ① 注册检查
try:
    T.plugin.autoload()
    report["autoload"] = "ok"
except Exception as e:                      # noqa: BLE001
    report["autoload"] = "fail: %s" % e
try:
    import polaris                          # noqa: F401  触发自动注册
    report["import_polaris"] = "ok"
except Exception as e:                      # noqa: BLE001
    report["import_polaris"] = "fail: %s" % e
cls = T.plugin.resolve_optimizer("Polaris")
report["resolve_Polaris"] = (cls.__name__ if cls is not None else None)
report["has_T_fit"] = hasattr(T, "fit")

# ② 任务
rng = np.random.default_rng(3)
D_IN, D_OUT, N = 128, 256, 8
x_np = rng.normal(size=(D_IN, N)).astype(np.float32)
t_np = rng.normal(size=(D_OUT, N)).astype(np.float32)


def make_params():
    w = T.tensor(rng2.normal(0, 0.02, (D_OUT, D_IN)).astype(np.float32),
                 requires_grad=True)
    return w, None


def run(make_opt, tag, steps=50):
    global rng2
    rng2 = np.random.default_rng(7)          # 每组相同初始化
    w, b = make_params()
    x = T.tensor(x_np)
    t = T.tensor(t_np)
    opt = make_opt([w])
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        y = T.matmul(w, x)
        d = y - t
        loss = (d * d).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        # ★ ORV 的张量没有 .item()；用运行时读标量（之前写成 hasattr 兜底 ⇒ 全是 NaN）
        losses.append(float(rt.read_scalar(loss._gpu)))
    rt.sync()
    dt = time.perf_counter() - t0
    tail = sum(losses[-10:]) / 10.0
    status = None
    if hasattr(opt, "status"):
        try:
            status = opt.status()
        except Exception as e:              # noqa: BLE001
            status = "status() 失败: %s" % e
    print("[%s] 首 %.4f 末段均值 %.6f  %.3f s  status=%s"
          % (tag, losses[0], tail, dt, json.dumps(status, ensure_ascii=False, default=str)),
          flush=True)
    return tail


report["adamw_tail"] = run(lambda ps: T.optim.AdamW(ps, lr=1e-2), "AdamW")
if cls is not None:
    report["polaris_tail"] = run(lambda ps: cls(ps, lr=1e-2), "Polaris")

print("[report] " + json.dumps(report, ensure_ascii=False, default=str), flush=True)
