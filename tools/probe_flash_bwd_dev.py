# -*- coding: utf-8 -*-
"""设备侧块化反向：实现 + 与 autograd 对拍（q/k/v 层面）。

## 要点

* 掩码用**有限哨兵** `NEG = -1e9`（`-inf` 会把 `sb - lse` 推成 `+inf` ⇒ NaN，上轮已踩）；
* `dQ` 每个 query 块独占，块内累加后 `T.cat(dim=1)` 拼回；
* `dK/dV` 跨 query 块累积 —— ORV **没有赋值索引**，所以每块存一份，最后沿新维求和；
* 全部设备侧，只有最后取一次 numpy。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")

import tgai_go.train as T  # noqa: E402

NEG = -1e9


def softmax_rows(S):
    m = S.max(-1, keepdims=True)
    e = np.exp(S - m)
    return e / e.sum(-1, keepdims=True)

def masked_lse(q, k, causal=True, scale=None):
    """主机侧算 LSE（只用于测试；生产应由前向保存）。"""
    D = q.shape[-1]
    scale = scale or 1.0 / np.sqrt(D)
    S = (q @ k.transpose(0, 1, 3, 2)) * scale
    if causal:
        n = S.shape[-1]
        m = np.tril(np.ones((S.shape[-2], n), bool))
        S = np.where(m, S, NEG)
    mx = S.max(-1, keepdims=True)
    lse = mx + np.log(np.maximum(np.exp(S - mx).sum(-1, keepdims=True), 1e-20))
    return lse[..., 0]


def flash_bwd_device(q, k, v, dO, lse, O, causal=True, cq=64, ck=64):
    """返回 (dq, dk, dv) 的 numpy；q,k,v,dO,O 形状 (B,H,S,D)。

    `O` 必须传入：softmax 分母修正项 `Σ_j P_ij dP_ij = dO_i · O_i` 要**跨所有 key**，
    块内求和是错的（本文件 2026-11-13 实测：只在 `ck < S` 时暴露，误差 ~3e-3）。
    """
    B, H, S, D = q.shape
    BH = B * H
    scale = 1.0 / np.sqrt(float(D))
    qt = T.tensor(q.reshape(BH, S, D))
    kt = T.tensor(k.reshape(BH, S, D))
    vt = T.tensor(v.reshape(BH, S, D))
    dot = T.tensor(dO.reshape(BH, S, D))
    lse_t = T.tensor(lse.reshape(BH, S))
    o_t = T.tensor(np.ascontiguousarray(O.reshape(BH, S, D), np.float32))

    dq_parts = []
    dk_parts = [[] for _ in range((S + ck - 1) // ck)]
    dv_parts = [[] for _ in range((S + ck - 1) // ck)]

    for qi in range(0, S, cq):
        rows = min(cq, S - qi)
        qb = qt[:, qi:qi + rows, :]
        db = dot[:, qi:qi + rows, :]
        lb = lse_t[:, qi:qi + rows]
        # ★ 切片是**非连续视图**；实测 `ck<S`（多个 key 块）时结果才错，
        #   嫌疑是"切片 → transpose → matmul"这条链。先 reshape 成连续，规避视图布局问题。
        qb = qb.reshape(BH, rows, D)
        db = db.reshape(BH, rows, D)
        # ★ 分母修正项一次算完：Σ_j P_ij dP_ij = dO_i · O_i（与 key 块无关）
        ob = o_t[:, qi:qi + rows, :].reshape(BH, rows, D)
        dsum = (db * ob).sum(axis=-1).reshape(BH, rows, 1)
        dq_b = T.tensor(np.zeros((BH, rows, D), np.float32))
        kmax = min(S, qi + rows) if causal else S
        for ki in range(0, kmax, ck):
            # ★ 满宽：cols 只由 S 决定，不由 kmax 截断 —— 否则同一 key 块在不同 query 块下宽度不同，
            #   dK/dV 的贡献形状不一致（实测 np.stack 直接报错），整除时则会暗中错位。
            cols = min(ck, S - ki)
            kb = kt[:, ki:ki + cols, :].reshape(BH, cols, D)
            vb = vt[:, ki:ki + cols, :].reshape(BH, cols, D)
            sb = (qb @ kb.transpose(1, 2)) * scale
            if causal:
                r = np.arange(rows)[:, None]
                cc = np.arange(cols)[None, :]
                arr = np.where((cc - r) <= (qi - ki), 0.0, NEG).astype(np.float32)
                sb = sb + T.tensor(arr.reshape(1, rows, cols))
            # P = exp(S - LSE)
            p = (sb - lb.reshape(BH, rows, 1)).exp()
            dv_parts[ki // ck].append(p.transpose(1, 2) @ db)
            dp = db @ vb.transpose(1, 2)
            ds = p * (dp - dsum)
            dq_b = dq_b + (ds @ kb) * scale
            dk_parts[ki // ck].append((ds.transpose(1, 2) @ qb) * scale)
        dq_parts.append(np.asarray(dq_b.to_numpy(), np.float32))

    dq = np.concatenate(dq_parts, axis=1)
    dk = np.zeros((BH, S, D), np.float32)
    dv = np.zeros((BH, S, D), np.float32)
    for bi in range(len(dk_parts)):
        if not dk_parts[bi]:
            continue
        ki = bi * ck
        cols = min(ck, S - ki)
        dk[:, ki:ki + cols, :] = np.sum(
            np.stack([np.asarray(x.to_numpy(), np.float32) for x in dk_parts[bi]], 0), 0)
        dv[:, ki:ki + cols, :] = np.sum(
            np.stack([np.asarray(x.to_numpy(), np.float32) for x in dv_parts[bi]], 0), 0)
    return (dq.reshape(B, H, S, D), dk.reshape(B, H, S, D), dv.reshape(B, H, S, D))


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    B, H, S, D = 1, 2, 24, 8
    q = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    k = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    v = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    dO = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)

    # ★ 关键对照：causal=False 时没有掩码边界，若它精确 ⇒ 问题就在掩码
    for causal in (False, True):
        qt = T.tensor(q, requires_grad=True)
        kt = T.tensor(k, requires_grad=True)
        vt = T.tensor(v, requires_grad=True)
        out = T.functional.scaled_dot_product_attention(qt, kt, vt, causal=causal)
        (out * T.tensor(dO)).sum().backward()
        gq = np.asarray(qt.grad.to_numpy(), np.float32)
        gk = np.asarray(kt.grad.to_numpy(), np.float32)
        gv = np.asarray(vt.grad.to_numpy(), np.float32)

        lse = masked_lse(q, k, causal=causal)
        Sq = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(D))
        if causal:
            n = Sq.shape[-1]
            Sq = np.where(np.tril(np.ones((Sq.shape[-2], n), bool)), Sq, NEG)
        Ofull = softmax_rows(Sq) @ v
        for (cq, ck) in ((S, S), (8, S), (S, 8), (8, 8)):
            dq, dk, dv = flash_bwd_device(q, k, v, dO, lse, Ofull, causal=causal, cq=cq, ck=ck)
            print("causal=%-5s cq=%-2d ck=%-2d dq=%.3e dk=%.3e dv=%.3e"
                  % (causal, cq, ck, float(np.abs(gq - dq).max()),
                     float(np.abs(gk - dk).max()), float(np.abs(gv - dv).max())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
