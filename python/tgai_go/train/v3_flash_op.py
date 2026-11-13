# -*- coding: utf-8 -*-
"""块化注意力算子（前向 + 反向重算）作为 ORV 自定义 VJP。

数值已在 `tools/probe_flash_bwd_dev.py` 上验过：八组 `(causal, cq, ck)` 组合误差 1e-9~1e-8。
本文件只是把它接成 ORV 的 autograd 节点（写法照 `scatter_add` / `_sdpa_lowmem`）。

要点（两条都是踩过的坑）：

1. 掩码用**有限哨兵** `NEG = -1e9`；`-inf` 会让 `sb - lse` 变成 `+inf` ⇒ NaN。
2. 反向的分母修正项是 `dO·O`（跨所有 key），**不能**用块内的 `(dP⊙P).sum(-1)`。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

NEG = -1e9


def _np(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


def _fwd_device(qt, kt, vt, causal, cq, ck):
    """返回 (O, LSE)，都是 numpy；q/k/v 是 (BH, S, D) 设备张量。"""
    import tgai_go.train as T
    BH, S, D = (int(s) for s in qt.shape)
    scale = 1.0 / np.sqrt(float(D))
    outs, lses = [], []
    for qi in range(0, S, cq):
        rows = min(cq, S - qi)
        qb = qt[:, qi:qi + rows, :].reshape(BH, rows, D)
        acc = T.tensor(np.zeros((BH, rows, D), np.float32))
        m = T.tensor(np.full((BH, rows), NEG, np.float32))
        l = T.tensor(np.zeros((BH, rows), np.float32))
        kmax = min(S, qi + rows) if causal else S
        for ki in range(0, kmax, ck):
            cols = min(ck, S - ki)
            kb = kt[:, ki:ki + cols, :].reshape(BH, cols, D)
            vb = vt[:, ki:ki + cols, :].reshape(BH, cols, D)
            sb = (qb @ kb.transpose(1, 2)) * scale
            if causal:
                r = np.arange(rows)[:, None]
                cc = np.arange(cols)[None, :]
                arr = np.where((cc - r) <= (qi - ki), 0.0, NEG).astype(np.float32)
                sb = sb + T.tensor(arr.reshape(1, rows, cols))
            m_new = T.cat([m.reshape(BH, rows, 1),
                           sb.max(axis=-1).reshape(BH, rows, 1)], dim=2).max(axis=-1)
            alpha = (m - m_new).exp()
            e = (sb - m_new.reshape(BH, rows, 1)).exp()
            l = l * alpha + e.sum(axis=-1)
            acc = acc * alpha.reshape(BH, rows, 1) + (e @ vb)
            m = m_new
        outs.append(_np(acc / l.reshape(BH, rows, 1)))
        lses.append(_np(m + l.log()))
    return np.concatenate(outs, axis=1), np.concatenate(lses, axis=1)


def _bwd_device(qt, kt, vt, dot, lse_t, ot, causal, cq, ck):
    """返回 (dq, dk, dv) 的 numpy。"""
    import tgai_go.train as T
    BH, S, D = (int(s) for s in qt.shape)
    scale = 1.0 / np.sqrt(float(D))
    dq_parts = []
    dk_parts = [[] for _ in range((S + ck - 1) // ck)]
    dv_parts = [[] for _ in range((S + ck - 1) // ck)]
    for qi in range(0, S, cq):
        rows = min(cq, S - qi)
        qb = qt[:, qi:qi + rows, :].reshape(BH, rows, D)
        db = dot[:, qi:qi + rows, :].reshape(BH, rows, D)
        ob = ot[:, qi:qi + rows, :].reshape(BH, rows, D)
        lb = lse_t[:, qi:qi + rows]
        dsum = (db * ob).sum(axis=-1).reshape(BH, rows, 1)   # ★ 跨所有 key 的分母修正
        dq_b = T.tensor(np.zeros((BH, rows, D), np.float32))
        kmax = min(S, qi + rows) if causal else S
        for ki in range(0, kmax, ck):
            cols = min(ck, S - ki)                            # ★ 满宽
            kb = kt[:, ki:ki + cols, :].reshape(BH, cols, D)
            vb = vt[:, ki:ki + cols, :].reshape(BH, cols, D)
            sb = (qb @ kb.transpose(1, 2)) * scale
            if causal:
                r = np.arange(rows)[:, None]
                cc = np.arange(cols)[None, :]
                arr = np.where((cc - r) <= (qi - ki), 0.0, NEG).astype(np.float32)
                sb = sb + T.tensor(arr.reshape(1, rows, cols))
            p = (sb - lb.reshape(BH, rows, 1)).exp()
            dv_parts[ki // ck].append(p.transpose(1, 2) @ db)
            dp = db @ vb.transpose(1, 2)
            ds = p * (dp - dsum)
            dq_b = dq_b + (ds @ kb) * scale
            dk_parts[ki // ck].append((ds.transpose(1, 2) @ qb) * scale)
        dq_parts.append(_np(dq_b))
    dq = np.concatenate(dq_parts, axis=1)
    dk = np.zeros((BH, S, D), np.float32)
    dv = np.zeros((BH, S, D), np.float32)
    for bi, parts in enumerate(dk_parts):
        if not parts:
            continue
        ki = bi * ck
        cols = min(ck, S - ki)
        dk[:, ki:ki + cols, :] = np.sum(np.stack([_np(x) for x in parts], 0), 0)
        dv[:, ki:ki + cols, :] = np.sum(np.stack([_np(x) for x in dv_parts[bi]], 0), 0)
    return dq, dk, dv


def flash_attn(q, k, v, causal=True, cq=64, ck=64):
    """`q,k,v` 形状 (B,H,S,D) 的 ORV 张量；返回 (B,H,S,D)。块化、不物化 S×S。"""
    import tgai_go.train as T

    B, H, S, D = (int(s) for s in q.shape)
    BH = B * H
    qt = q.reshape(BH, S, D)
    kt = k.reshape(BH, S, D)
    vt = v.reshape(BH, S, D)
    o_np, lse_np = _fwd_device(qt, kt, vt, causal, cq, ck)
    need = bool(q.requires_grad or k.requires_grad or v.requires_grad)

    buf = T.tensor(o_np.reshape(B, H, S, D))
    out = T.Tensor._from_gpu(buf._gpu, requires_grad=need,
                             _parents=(q, k, v) if need else ())
    if not need:
        return out

    lse_full = np.ascontiguousarray(lse_np.reshape(B, H, S))

    def bw(g, grads):
        from tgai_go.train import _acc
        dot = T.tensor(_np(g).reshape(BH, S, D))
        lse_t = T.tensor(lse_full.reshape(BH, S))
        ot = T.tensor(o_np.reshape(BH, S, D))
        dq, dk, dv = _bwd_device(qt, kt, vt, dot, lse_t, ot, causal, cq, ck)
        if q.requires_grad:
            _acc(grads, q, T.tensor(dq.reshape(B, H, S, D))._gpu)
        if k.requires_grad:
            _acc(grads, k, T.tensor(dk.reshape(B, H, S, D))._gpu)
        if v.requires_grad:
            _acc(grads, v, T.tensor(dv.reshape(B, H, S, D))._gpu)

    out._backward = bw
    return out
