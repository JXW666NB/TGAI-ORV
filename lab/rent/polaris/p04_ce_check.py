# -*- coding: utf-8 -*-
"""CE 替代实现校验：向量化 CE 与 ORV 内置 CPU CE 的 loss/grad 是否一致（float64 参考）。"""
import json
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import tgai_go.train as T

out = {}
rng = np.random.default_rng(0)
N, V = 64, 4096
logits_np = (rng.normal(0, 1.5, (N, V))).astype(np.float32)
tgt = rng.integers(0, V, N).astype(np.int64)


def ce_vec(logits, y):
    """向量化 CE（ORV 张量算子），logits [N,V] -> 标量 mean loss。

    行最大值用 numpy 常量（softmax 对常数平移不变，梯度不受影响），
    其余全走 ORV 张量算子 -> 反传自动。
    """
    l_np = np.asarray(logits.to_numpy(), np.float32)
    m = l_np.max(axis=1, keepdims=True)          # 常量，detach
    z = logits - m
    ez = T.exp(z)
    s = ez.sum(axis=1, keepdims=True)
    lp = z - T.log(s)
    idx = np.arange(int(logits.shape[0]))
    picked = lp[idx, y]
    return (-picked).mean()


def ce_ref64(logits, y):
    x = np.asarray(logits, np.float64)
    mx = x.max(axis=1, keepdims=True)
    z = x - mx
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
    lp = z - lse
    return float(-lp[np.arange(x.shape[0]), y].mean())


def grad_of(fn):
    w = T.tensor(logits_np.copy(), requires_grad=True)
    loss = fn(w)
    loss.backward()
    g = w.grad
    g = np.asarray(g.to_numpy() if hasattr(g, "to_numpy") else g, np.float64)
    return float(np.asarray(loss.to_numpy())), g


res = {}
try:
    l_builtin, g_builtin = grad_of(lambda w: T.functional.cross_entropy(w, tgt))
    res["builtin"] = {"loss": l_builtin}
except Exception as e:
    res["builtin_err"] = "%s: %s" % (type(e).__name__, e)
try:
    l_vec, g_vec = grad_of(lambda w: ce_vec(w, tgt))
    res["vectorized"] = {"loss": l_vec}
except Exception as e:
    res["vec_err"] = "%s: %s" % (type(e).__name__, e)
res["float64_ref"] = ce_ref64(logits_np, tgt)

# float64 解析梯度
def grad_ref64(logits, y):
    x = np.asarray(logits, np.float64)
    mx = x.max(axis=1, keepdims=True)
    z = x - mx
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    g = p.copy()
    g[np.arange(x.shape[0]), y] -= 1.0
    return g / x.shape[0]


gref = grad_ref64(logits_np, tgt)
if "builtin" in res:
    res["builtin"]["max_abs_grad_err_vs_f64"] = float(np.abs(g_builtin - gref).max())
    res["builtin"]["rel_fro_err"] = float(np.linalg.norm(g_builtin - gref) / np.linalg.norm(gref))
if "vectorized" in res:
    res["vectorized"]["max_abs_grad_err_vs_f64"] = float(np.abs(g_vec - gref).max())
    res["vectorized"]["rel_fro_err"] = float(np.linalg.norm(g_vec - gref) / np.linalg.norm(gref))
if "builtin" in res and "vectorized" in res:
    res["vec_vs_builtin"] = {
        "loss_diff": abs(res["vectorized"]["loss"] - res["builtin"]["loss"]),
        "max_abs_grad_diff": float(np.abs(g_vec - g_builtin).max()),
    }

# 计时
for name, fn in (("builtin", lambda: T.functional.cross_entropy(
        T.tensor(logits_np.copy()), tgt)),
                 ("vectorized", lambda: ce_vec(T.tensor(logits_np.copy()), tgt))):
    try:
        t0 = time.perf_counter()
        fn()
        res["time_%s_s" % name] = round(time.perf_counter() - t0, 4)
    except Exception as e:
        res["time_%s_err" % name] = str(e)

# backward+step 稳态计时（AdamW，排除首次 import 开销）
model = T.preset.transformer(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512)
ps = list(model.parameters())
f = np.load(r"D:\orv_bench\data2048_v4096\shard_000.npz")
ids = np.asarray(f["input_ids"]).reshape(-1)
seq = 64
x = ids[:seq].astype(np.int64)
y = ids[1:seq + 1].astype(np.int64)


def one_step(opt):
    logits = model(x)                      # [1,64,4096]
    L = logits.reshape(seq, 4096)
    loss = ce_vec(L, y)
    for p in ps:
        p.grad = None
    loss.backward()
    opt.step()
    return float(np.asarray(loss.to_numpy()))


steps_t = {}
for nm, opt in (("AdamW", T.optim.AdamW(ps, lr=1e-3)),
                ("Adafactor", T.optim.Adafactor(ps, lr=1e-3)),
                ("LAMB", T.optim.LAMB(ps, lr=1e-3))):
    ts = []
    for i in range(4):
        t0 = time.perf_counter()
        lv = one_step(opt)
        ts.append(time.perf_counter() - t0)
    steps_t[nm] = {"loss": lv, "s": [round(t, 4) for t in ts]}
try:
    import polaris  # noqa: F401
    P = T.plugin.resolve_optimizer("Polaris")
    o = P(ps, lr=1e-3)
    ts = []
    for i in range(4):
        t0 = time.perf_counter()
        lv = one_step(o)
        ts.append(time.perf_counter() - t0)
    steps_t["Polaris"] = {"loss": lv, "s": [round(t, 4) for t in ts],
                          "status": o.status()}
except Exception as e:
    steps_t["Polaris"] = "ERR:%s: %s" % (type(e).__name__, e)
res["step_times"] = steps_t

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "p04_ce_check.json")
with open(p, "w", encoding="utf-8") as fp:
    json.dump(res, fp, ensure_ascii=False, indent=1, default=str)
print("WROTE " + p)
print(json.dumps(res, ensure_ascii=True, indent=1, default=str))
