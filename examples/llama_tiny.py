# -*- coding: utf-8 -*-
"""试金石（CPU 版）：LLaMA 结构用 tgai_go.train 训练，并与 PyTorch 孪生逐步对齐。

## 这个脚本回答什么问题

> 别人的模型结构（RMSNorm + RoPE + SwiGLU + **GQA** + 无 bias），
> 我们的引擎能不能**训**，而且和 PyTorch 走出同样的曲线？

它**不依赖 CUDA 内核**：全部用 `tgai_go.train` 的张量算子 + `nn_ops` 组合而成，
所以在本机 CPU 上就能跑，也能进 CI（不需要 GPU）。

## 判据

* 每一步 loss 与 PyTorch 孪生的相对偏差 ≤ 1e-3；
* 训练后参数相对偏差 ≤ 1e-3；
* 20 步 loss 必须下降（证明梯度确实在起效）。

用法：
    python examples/llama_tiny.py --steps 20
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)

import tgai_go.train as T  # noqa: E402
from tgai_go.train import functional as F  # noqa: E402
from tgai_go.train import nn_ops as O  # noqa: E402

CFG = dict(V=64, D=64, L=2, NH=4, KV=2, FF=176, S=16, B=4, THETA=10000.0, EPS=1e-6)

#: 尺寸预设：tiny 用于 CI，m60 是目标里的「60M 级」LLaMA 结构
PRESETS = {
    "tiny": dict(V=64, D=64, L=2, NH=4, KV=2, FF=176, S=16, B=4),
    # 60M 级：D=768, L=8, FF=2048, V=8192, GQA 8:2 ⇒ 约 63M 参数
    "m60": dict(V=8192, D=768, L=8, NH=8, KV=2, FF=2048, S=64, B=2),
}


def init_weights(cfg, seed=1234):
    """统一的初始化：两边用**同一组 numpy 权重**，保证是同一只模型。"""
    rng = np.random.default_rng(seed)
    V, D, L, NH, KV, FF = cfg["V"], cfg["D"], cfg["L"], cfg["NH"], cfg["KV"], cfg["FF"]
    DH = D // NH
    w = {"emb": rng.normal(0, 0.02, (V, D)).astype(np.float32),
         "norm": np.ones(D, np.float32)}
    for i in range(L):
        p = "l%d." % i
        # 统一按 [out, in] 存（与 nn_ops.linear / torch.nn.Linear 一致）
        for nm, shape in (("wq", (NH * DH, D)), ("wk", (KV * DH, D)), ("wv", (KV * DH, D)),
                          ("wo", (D, NH * DH)), ("w1", (FF, D)), ("w2", (FF, D)),
                          ("w3", (D, FF))):
            w[p + nm] = (rng.normal(0, 0.02, shape)).astype(np.float32)
        w[p + "ln1"] = np.ones(D, np.float32)
        w[p + "ln2"] = np.ones(D, np.float32)
    return w


def rope_tables(seq, dh, theta):
    inv = 1.0 / (theta ** (np.arange(0, dh, 2, dtype=np.float64) / dh))
    ang = np.outer(np.arange(seq, dtype=np.float64), inv)
    c = np.cos(ang).astype(np.float32)
    s = np.sin(ang).astype(np.float32)
    return np.concatenate([c, c], axis=-1), np.concatenate([s, s], axis=-1)


def _as(w, key):
    """权重可能是 numpy（首次）也可能是已包装的 Tensor（训练时）。"""
    v = w[key]
    return v if isinstance(v, T.Tensor) else T.tensor(np.ascontiguousarray(v))


def ours_forward(w, ids, cfg):
    """用 tgai_go.train 的算子搭 LLaMA 前向。"""
    D, L, NH, KV, FF, EPS = cfg["D"], cfg["L"], cfg["NH"], cfg["KV"], cfg["FF"], cfg["EPS"]
    DH = D // NH
    seq = len(ids)
    x = O.embedding(_as(w, "emb"), np.asarray(ids, dtype=np.int64))   # [S,D]
    cos, sin = rope_tables(seq, DH, cfg["THETA"])
    cos_t = T.tensor(cos[None, :, :])
    sin_t = T.tensor(sin[None, :, :])
    mask = T.tensor(np.triu(np.full((seq, seq), -1e9, np.float32), k=1))

    def apply_rope(t):
        """t: [H,S,DH]"""
        half = DH // 2
        x1, x2 = t[:, :, :half], t[:, :, half:]
        rot = T.cat([-x2, x1], dim=-1)
        return t * cos_t + rot * sin_t

    for i in range(L):
        p = "l%d." % i
        h = O.rms_norm(x, _as(w, p + "ln1"), EPS)
        q = O.linear(h, _as(w, p + "wq")).reshape(seq, NH, DH).permute(1, 0, 2)
        k = O.linear(h, _as(w, p + "wk")).reshape(seq, KV, DH).permute(1, 0, 2)
        v = O.linear(h, _as(w, p + "wv")).reshape(seq, KV, DH).permute(1, 0, 2)
        q, k = apply_rope(q), apply_rope(k)
        k = k.repeat_interleave(NH // KV, dim=0)
        v = v.repeat_interleave(NH // KV, dim=0)
        att = F.softmax((q @ k.swap_last2()) * (DH ** -0.5) + mask, -1)
        o = (att @ v).permute(1, 0, 2).reshape(seq, D)
        x = x + O.linear(o, _as(w, p + "wo"))
        h2 = O.rms_norm(x, _as(w, p + "ln2"), EPS)
        g = O.linear(h2, _as(w, p + "w1"))
        u = O.linear(h2, _as(w, p + "w2"))
        x = x + O.linear(F.silu(g) * u, _as(w, p + "w3"))
    x = O.rms_norm(x, _as(w, "norm"), EPS)
    return O.linear(x, _as(w, "emb"))       # 权重绑定 lm_head


def _pairs_from_ids(ids_all, seq, n=4):
    """从长 id 序列切出 n 个 (输入, 目标) 窗口。"""
    out = []
    span = seq + 1
    for k in range(n):
        st = k * span
        if st + span > len(ids_all):
            break
        win = ids_all[st:st + span]
        out.append((win[:-1], win[1:]))
    return out


def ours_train(w, ids, tgt, steps, lr, cfg):
    params = {}
    for k, v in w.items():
        params[k] = T.tensor(v.copy(), requires_grad=True)
    pairs = (ids if (isinstance(ids, list) and ids and isinstance(ids[0], tuple))
             else [(ids, tgt)])
    losses, state = [], None
    for step in range(steps):
        xb, yb = pairs[step % len(pairs)]
        for p in params.values():
            p.zero_grad()
        logits = ours_forward(params, xb, cfg)
        loss = O.cross_entropy(logits.reshape(len(xb), cfg["V"]), yb)
        loss.backward()
        state = O.sgd_step(list(params.values()), lr=lr, momentum=0.9, state=state)
        losses.append(float(loss.to_numpy()))
    return losses, {k: p.to_numpy() for k, p in params.items()}


def torch_train(w, ids, tgt, steps, lr, cfg, torch):
    """PyTorch 孪生：同样的权重、同样的结构、同样的优化器。"""
    import torch.nn.functional as TF
    D, L, NH, KV, FF, EPS = cfg["D"], cfg["L"], cfg["NH"], cfg["KV"], cfg["FF"], cfg["EPS"]
    DH = D // NH
    seq = len(ids)
    params = {k: torch.nn.Parameter(torch.from_numpy(v.copy())) for k, v in w.items()}
    opt = torch.optim.SGD(list(params.values()), lr=lr, momentum=0.9)
    # 支持「多窗口轮转」：序列长度从第一对窗口取（我们的窗口等长）
    pairs = (ids if (isinstance(ids, list) and ids and isinstance(ids[0], tuple))
             else [(ids, tgt)])
    seq = int(np.asarray(pairs[0][0]).reshape(-1).shape[0])
    cos, sin = rope_tables(seq, DH, cfg["THETA"])
    c = torch.from_numpy(cos)[None]
    s = torch.from_numpy(sin)[None]
    m = torch.from_numpy(np.triu(np.full((seq, seq), -1e9, np.float32), k=1))

    def apply_rope(t):
        half = DH // 2
        x1, x2 = t[..., :half], t[..., half:]
        rot = torch.cat([-x2, x1], dim=-1)
        return t * c + rot * s

    losses = []
    for step in range(steps):
        xb, yb = pairs[step % len(pairs)]
        opt.zero_grad()
        x = params["emb"][torch.from_numpy(xb)]
        for i in range(L):
            p = "l%d." % i
            h = TF.rms_norm(x, (D,), params[p + "ln1"], EPS)
            q = TF.linear(h, params[p + "wq"]).reshape(seq, NH, DH).permute(1, 0, 2)
            k = TF.linear(h, params[p + "wk"]).reshape(seq, KV, DH).permute(1, 0, 2)
            v = TF.linear(h, params[p + "wv"]).reshape(seq, KV, DH).permute(1, 0, 2)
            q, k = apply_rope(q), apply_rope(k)
            k = k.repeat_interleave(NH // KV, dim=0)
            v = v.repeat_interleave(NH // KV, dim=0)
            att = torch.softmax((q @ k.transpose(-1, -2)) * (DH ** -0.5) + m, -1)
            o = (att @ v).permute(1, 0, 2).reshape(seq, D)
            x = x + TF.linear(o, params[p + "wo"])
            h2 = TF.rms_norm(x, (D,), params[p + "ln2"], EPS)
            g = TF.linear(h2, params[p + "w1"])
            u = TF.linear(h2, params[p + "w2"])
            x = x + TF.linear(TF.silu(g) * u, params[p + "w3"])
        x = TF.rms_norm(x, (D,), params["norm"], EPS)
        logits = x @ params["emb"].T
        loss = TF.cross_entropy(logits, torch.from_numpy(yb))
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses, {k: p.detach().numpy() for k, p in params.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    ap.add_argument("--corpus", default="", help="用真实语料（字符级词表）代替随机 token")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tol-loss", type=float, default=1e-3)
    ap.add_argument("--tol-param", type=float, default=1e-3)
    a = ap.parse_args(argv)

    T.set_default_device("cpu")
    cfg = dict(CFG)
    cfg.update(PRESETS[a.preset])
    if a.corpus:
        text = open(a.corpus, encoding="utf-8", errors="replace").read()
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        cfg = dict(cfg)
        cfg["V"] = len(chars)
        need = cfg["S"] * 8 + 8
        ids_all = np.array([stoi[c] for c in text[:max(need, 4096)]], dtype=np.int64)
        pairs = _pairs_from_ids(ids_all, cfg["S"], n=4)
        ids = [(p[0], p[1]) for p in pairs]
        print("语料：%s（%d 字符，词表 %d，切出 %d 个窗口）"
              % (os.path.basename(a.corpus), len(text), len(chars), len(pairs)))
        tgt = None
    else:
        rng = np.random.default_rng(7)
        ids = rng.integers(0, cfg["V"], size=cfg["S"]).astype(np.int64)
        tgt = rng.integers(0, cfg["V"], size=cfg["S"]).astype(np.int64)
    w = init_weights(cfg)

    print("配置：D=%d L=%d NH=%d KV=%d DH=%d FF=%d S=%d V=%d steps=%d lr=%g"
          % (cfg["D"], cfg["L"], cfg["NH"], cfg["KV"], cfg["D"] // cfg["NH"],
             cfg["FF"], cfg["S"], cfg["V"], a.steps, a.lr))
    print("结构：RMSNorm + RoPE + SwiGLU + GQA(%d:%d) + 无 bias + 权重绑定" % (cfg["NH"], cfg["KV"]))
    n_par = sum(v.size for v in w.values())
    print("参数量：%.2fM" % (n_par / 1e6))

    ours_loss, ours_w = ours_train(w, ids, tgt, a.steps, a.lr, cfg)
    try:
        import torch
    except ImportError:
        print("\n⚠ 没装 torch：只跑我方训练，无法做孪生对比")
        print("  我们 loss: 首 %.6f → 末 %.6f" % (ours_loss[0], ours_loss[-1]))
        return 0
    ref_loss, ref_w = torch_train(w, ids, tgt, a.steps, a.lr, cfg, torch)

    print("\n  step |      ours |     torch |  相对偏差")
    print("  -----+-----------+-----------+----------")
    worst = 0.0
    for i, (o, r) in enumerate(zip(ours_loss, ref_loss)):
        rel = abs(o - r) / max(abs(r), 1e-9)
        worst = max(worst, rel)
        if i < 5 or i >= a.steps - 3:
            print("  %4d | %9.6f | %9.6f | %8.2e" % (i + 1, o, r, rel))
        elif i == 5:
            print("   ... |       ... |       ... |      ...")

    param_rel = max(
        float(np.abs(ours_w[k].astype(np.float64) - ref_w[k].astype(np.float64)).max()
              / (np.abs(ref_w[k]).max() + 1e-12)) for k in ours_w)
    drop = (ours_loss[0] - ours_loss[-1]) / max(ours_loss[0], 1e-9)

    print("")
    print("loss 首末：ours %.6f → %.6f（下降 %.1f%%），torch %.6f → %.6f"
          % (ours_loss[0], ours_loss[-1], 100 * drop, ref_loss[0], ref_loss[-1]))
    print("最大逐步相对偏差：%.2e（判据 ≤ %.0e）" % (worst, a.tol_loss))
    print("训练后参数相对偏差：%.2e（判据 ≤ %.0e）" % (param_rel, a.tol_param))
    ok = worst <= a.tol_loss and param_rel <= a.tol_param and drop > 0
    print("")
    print("结论：%s" % ("PASS —— 别人的结构我们能训，且与 PyTorch 同步" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
