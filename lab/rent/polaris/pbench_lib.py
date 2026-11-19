# -*- coding: utf-8 -*-
"""Polaris 基准测试公共库（ORV CPU/CUDA 通用）。

口径（全矩阵统一）：
  * 无梯度裁剪、无 lr 调度、无 weight decay —— 只比"优化器更新规则"本身
  * 每个 (优化器, lr) 用同一份初始化与同一份数据顺序
  * 末段 = 最后 10 步的算术均值；min = 全程最小 loss
  * 步时 = 第 warm 步之后的平均（包含 fwd+bwd+opt.step 的整步）
"""
import json
import math
import os
import sys
import time

import numpy as np


def utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def out_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(d, exist_ok=True)
    return d


def dump(name, obj):
    p = os.path.join(out_dir(), name + ".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    print("WROTE " + p)
    return p


def load(name):
    p = os.path.join(out_dir(), name + ".json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 任务 1：玩具回归（有唯一 float64 最优点）
# ---------------------------------------------------------------------------
TOY = dict(d_in=32, d_out=64, n=256, noise=0.1, steps=1200, seed=0, init_std=0.02)


def toy_problem(cfg=None):
    c = dict(TOY)
    if cfg:
        c.update(cfg)
    rng = np.random.default_rng(12345)              # 数据固定，不随 seed 变
    X = rng.normal(0, 1, (c["d_in"], c["n"])).astype(np.float64)
    W_true = rng.normal(0, 1.0 / math.sqrt(c["d_in"]),
                        (c["d_out"], c["d_in"])).astype(np.float64)
    noise = rng.normal(0, c["noise"], (c["d_out"], c["n"])).astype(np.float64)
    Tt = W_true @ X + noise
    # float64 闭式最优点（最小二乘）
    W_star = np.linalg.lstsq(X.T, Tt.T, rcond=None)[0].T     # (d_out, d_in)
    resid = W_star @ X - Tt
    loss_star = float((resid * resid).mean())
    W0 = np.random.default_rng(1000 + c["seed"]).normal(
        0, c["init_std"], (c["d_out"], c["d_in"])).astype(np.float32)
    return c, X.astype(np.float32), Tt.astype(np.float32), W0, W_star, loss_star


# ---------------------------------------------------------------------------
# 任务 2：迷你 V3 模型（真 Transformer，拿 data2048_v4096 的 token）
# ---------------------------------------------------------------------------
MODEL = dict(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512,
             seq=64, steps=200, seed=0)
DATA_V4096 = r"D:\orv_bench\data2048_v4096"


def model_blocks(seq, n_blocks, shard_dir=DATA_V4096):
    """按固定顺序切出 n_blocks 个 (x, y)，跨 shard 顺序读取。"""
    xs, ys = [], []
    shards = sorted(f for f in os.listdir(shard_dir) if f.endswith(".npz"))
    for sh in shards:
        f = np.load(os.path.join(shard_dir, sh))
        ids = np.asarray(f["input_ids"]).reshape(-1).astype(np.int64)
        k = 0
        while k + seq + 1 <= ids.size and len(xs) < n_blocks:
            seg = ids[k:k + seq + 1]
            xs.append(np.ascontiguousarray(seg[:-1]))
            ys.append(np.ascontiguousarray(seg[1:]))
            k += seq
        if len(xs) >= n_blocks:
            break
    return xs, ys


def ce_vec(T, logits, y):
    """向量化 CE：logits [N,V] -> 标量 mean loss（行最大值当常量，梯度不受影响）。

    CPU/CUDA 同一实现：`lp[np.arange, y]` 在 CUDA 上不被支持（0.9.97 的
    `_slice_gather_plan` 只接受标量/切片），所以改用 one-hot 乘 + sum。
    """
    l_np = np.asarray(logits.to_numpy(), np.float32)
    n, v = int(l_np.shape[0]), int(l_np.shape[1])
    m = l_np.max(axis=1, keepdims=True)
    z = logits - m
    ez = T.exp(z)
    s = ez.sum(axis=1, keepdims=True)
    lp = z - T.log(s)
    oh = np.zeros((n, v), np.float32)
    oh[np.arange(n), np.asarray(y).reshape(-1)] = 1.0
    picked = T._as_tensor(oh) if hasattr(T, "_as_tensor") else None
    prod = lp * oh if picked is None else lp * picked
    return -(prod.sum(axis=1)).sum() * (1.0 / float(n))


# ---------------------------------------------------------------------------
# 通用：跑一次
# ---------------------------------------------------------------------------
def make_opt(T, name, params, lr, **kw):
    """统一构造（每个优化器用它自己的默认 betas / eps / momentum）。"""
    if name == "AdamW":
        return T.optim.AdamW(params, lr=lr, weight_decay=0.0, **kw)
    if name == "Adafactor":
        return T.optim.Adafactor(params, lr=lr, weight_decay=0.0, **kw)
    if name == "Muon":
        return T.optim.Muon(params, lr=lr, weight_decay=0.0, **kw)
    if name == "LAMB":
        return T.optim.LAMB(params, lr=lr, weight_decay=0.0, **kw)
    if name == "Lion":
        return T.optim.Lion(params, lr=lr, weight_decay=0.0, **kw)
    if name == "SGD":
        return T.optim.SGD(params, lr=lr, weight_decay=0.0, momentum=0.9, **kw)
    if name == "Polaris":
        P = T.plugin.resolve_optimizer("Polaris")
        return P(params, lr=lr, **kw)
    raise KeyError(name)


def summarize(curve, warm=20, t_steps=None):
    c = [x for x in curve if x == x]                 # 去掉 nan
    finite = bool(len(c) == len(curve))
    tail = [x for x in curve[-10:] if x == x]
    r = {
        "steps_done": len(curve),
        "first": curve[0] if curve else None,
        "final": curve[-1] if curve else None,
        "min": min(c) if c else None,
        "last10_mean": (sum(tail) / len(tail)) if tail else None,
        "last10_n": len(tail),
        "all_finite": finite,
        "nan_step": (next((i for i, x in enumerate(curve) if x != x), None)
                     if not finite else None),
    }
    if t_steps:
        st = [t for t in t_steps[warm:] if t == t]
        r["t_step_s"] = (sum(st) / len(st)) if st else None
        r["t_step_n"] = len(st)
        r["t_total_s"] = sum(t for t in t_steps if t == t)
    # 固定检查点（便于跨优化器比较学习速度）
    for k in (10, 25, 50, 100, 200, 400):
        if len(curve) >= k:
            r["loss_at_%d" % k] = curve[k - 1]
    return r


def steps_to(curve, thr):
    """第一次 loss <= thr 的步数（1-based）；没到返回 None。"""
    for i, v in enumerate(curve):
        if v == v and v <= thr:
            return i + 1
    return None


def run_toy(T, name, lr, seed=0, steps=None, cfg=None, opt_kw=None, lr_sched=None):
    c, X, Tt, W0, W_star, loss_star = toy_problem(dict(cfg or {}, seed=seed))
    steps = int(steps or c["steps"])
    W = T.tensor(W0.copy(), requires_grad=True)
    x = T.tensor(X)
    t = T.tensor(Tt)
    opt = make_opt(T, name, [W], lr, **(opt_kw or {}))
    curve, ts = [], []
    for i in range(steps):
        if lr_sched is not None:
            opt.lr = lr_sched(i, steps, lr)
        t0 = time.perf_counter()
        d = T.matmul(W, x) - t
        loss = (d * d).sum() * (1.0 / float(np.prod(d.shape)))     # == mean，逐位等价
        W.grad = None
        loss.backward()
        opt.step()
        curve.append(float(np.asarray(loss.to_numpy())))
        ts.append(time.perf_counter() - t0)
    Wf = np.asarray(W.to_numpy(), np.float64)
    r = summarize(curve, t_steps=ts)
    r.update({
        "task": "toy", "opt": name, "lr": lr, "seed": seed, "steps": steps,
        "loss_star_f64": loss_star,
        "excess_final": (r["final"] - loss_star) if r["final"] is not None else None,
        "excess_last10": (r["last10_mean"] - loss_star) if r["last10_mean"] is not None else None,
        "rel_dist_to_Wstar": float(np.linalg.norm(Wf - W_star) / np.linalg.norm(W_star)),
        "steps_to_excess_1e-2": steps_to(curve, loss_star + 1e-2),
        "steps_to_excess_1e-3": steps_to(curve, loss_star + 1e-3),
        "steps_to_excess_1e-5": steps_to(curve, loss_star + 1e-5),
        "curve": [round(v, 10) if v == v else None for v in curve],
    })
    if name == "Polaris" and hasattr(opt, "status"):
        r["opt_status"] = opt.status()
    return r


def run_model(T, name, lr, seed=0, steps=None, cfg=None, opt_kw=None,
              xs=None, ys=None, lr_sched=None):
    c = dict(MODEL)
    c.update(cfg or {})
    c["seed"] = seed
    steps = int(steps or c["steps"])
    np.random.seed(1000 + seed)                      # ORV 初始化走 numpy 全局种子
    T.manual_seed(1000 + seed) if hasattr(T, "manual_seed") else None
    model = T.preset.transformer(vocab=c["vocab"], d_model=c["d_model"],
                                 n_layers=c["n_layers"], n_heads=c["n_heads"],
                                 d_ff=c["d_ff"])
    params = list(model.parameters())
    if xs is None or ys is None:
        xs, ys = model_blocks(c["seq"], steps)
    opt = make_opt(T, name, params, lr, **(opt_kw or {}))
    curve, ts = [], []
    for i in range(steps):
        x, y = xs[i % len(xs)], ys[i % len(ys)]
        if lr_sched is not None:
            opt.lr = lr_sched(i, steps, lr)
        t0 = time.perf_counter()
        logits = model(x)
        loss = ce_vec(T, logits, y)
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
        curve.append(float(np.asarray(loss.to_numpy())))
        ts.append(time.perf_counter() - t0)
    r = summarize(curve, t_steps=ts)
    r.update({
        "task": "mini_v3", "opt": name, "lr": lr, "seed": seed, "steps": steps,
        "cfg": c, "n_params": int(model.n_params()),
        "tokens_per_step": int(c["seq"]),
        "tokens_per_s": (c["seq"] / r["t_step_s"]) if r.get("t_step_s") else None,
        "curve": [round(v, 6) if v == v else None for v in curve],
    })
    if name == "Polaris" and hasattr(opt, "status"):
        r["opt_status"] = opt.status()
        try:
            sm = float(np.abs(np.asarray(params[0].data)).max())
            r["opt_status"]["w0_absmax"] = sm
        except Exception:
            pass
    return r


# ---------------------------------------------------------------------------
# lr 网格
# ---------------------------------------------------------------------------
GRID = {
    "AdamW": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 6e-2],
    "Adafactor": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    "Muon": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    "LAMB": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "Lion": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "Polaris": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
}


def table(records, key="last10_mean"):
    """按优化器汇总：最优 lr（按 key 最小）。"""
    best = {}
    for r in records:
        if r.get(key) is None or not r.get("all_finite", True):
            continue
        if r["opt"] not in best or r[key] < best[r["opt"]][key]:
            best[r["opt"]] = r
    return best
