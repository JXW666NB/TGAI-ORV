# -*- coding: utf-8 -*-
"""设备侧块化在线 softmax（v3）：修掉 NaN 的根因 —— 掩码哨兵值必须**有限**。

## 根因（实测定位）

掩码用 `-inf`、初始运行最大值 `m = -1e30`。整行被掩时 `m_new = max(-1e30, -inf) = -1e30`（有限），
于是

    sb − m_new = −inf − (−1e30) = +inf  ⇒  exp(+inf) = inf  ⇒  l = inf, acc = inf → inf/inf = NaN

主机版靠着 `np.where(np.isfinite(sb), e, 0)` 才没炸；设备版丢了这层保护。

## 修法

掩码用**有限**大负数 `NEG = -1e9`（不用 `-inf`）：被掩位置 `sb − m_new ≈ -1e9` ⇒ `exp ≈ 0`，
既不产生 `inf`，也不需要逐元素 `isfinite` 保护。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

NEG = -1e9          # ★ 有限哨兵：-inf 会把 sb - m_new 推成 +inf


def attention_flash_fwd_device(x, wq, wk, wv, wo, cos, sin, rot_dim, n_q, n_kv,
                               cq=64, ck=64, causal=True):
    import tgai_go.train as T
    from tgai_go.train import v3_ops as V

    B = int(x.shape[0]); Tlen = int(x.shape[1]); n_q = int(n_q); n_kv = int(n_kv)
    d_k = int(wq.shape[0]) // n_q
    scale = 1.0 / np.sqrt(float(d_k))

    with T.no_grad():
        q = (x @ wq.transpose(0, 1)).reshape(B, Tlen, n_q, d_k).transpose(1, 2)
        k = (x @ wk.transpose(0, 1)).reshape(B, Tlen, n_kv, d_k).transpose(1, 2)
        v = (x @ wv.transpose(0, 1)).reshape(B, Tlen, n_kv, d_k).transpose(1, 2)
        c = cos.reshape(1, 1, Tlen, int(rot_dim)); s = sin.reshape(1, 1, Tlen, int(rot_dim))
        q = V.v3_rope(q, c, s); k = V.v3_rope(k, c, s)
        if n_kv != n_q:
            rep = n_q // n_kv
            k = T.cat([k[:, h:h + 1] for h in range(n_kv) for _ in range(rep)], dim=1)
            v = T.cat([v[:, h:h + 1] for h in range(n_kv) for _ in range(rep)], dim=1)

        cache = {}

        def mask_t(rows, cols, delta):
            key = (rows, cols, delta)
            if key not in cache:
                r = np.arange(rows)[:, None]
                cc = np.arange(cols)[None, :]
                arr = np.where((cc - r) <= delta, 0.0, NEG).astype(np.float32)
                cache[key] = T.tensor(arr.reshape(1, rows, cols))
            return cache[key]

        outs, lses = [], []
        for qi in range(0, Tlen, cq):
            rows = min(cq, Tlen - qi)
            qb = q[:, :, qi:qi + rows, :].reshape(B * n_q, rows, d_k)
            acc = T.tensor(np.zeros((B * n_q, rows, d_k), np.float32))
            m = T.tensor(np.full((B * n_q, rows), NEG, np.float32))
            l = T.tensor(np.zeros((B * n_q, rows), np.float32))
            kmax = min(Tlen, qi + rows) if causal else Tlen
            for ki in range(0, kmax, ck):
                cols = min(ck, kmax - ki)
                kb = k[:, :, ki:ki + cols, :].reshape(B * n_q, cols, d_k)
                vb = v[:, :, ki:ki + cols, :].reshape(B * n_q, cols, d_k)
                sb = (qb @ kb.transpose(1, 2)) * scale
                if causal:
                    delta = qi - ki
                    if delta < cols - 1 or delta < rows - 1:
                        sb = sb + mask_t(rows, cols, delta)
                m_new = T.cat([m.reshape(B * n_q, rows, 1),
                               sb.max(axis=-1).reshape(B * n_q, rows, 1)],
                              dim=2).max(axis=-1)
                alpha = (m - m_new).exp()
                e = (sb - m_new.reshape(B * n_q, rows, 1)).exp()
                l = l * alpha + e.sum(axis=-1)
                acc = acc * alpha.reshape(B * n_q, rows, 1) + (e @ vb)
                m = m_new
            lse = m + l.log()
            outs.append(np.asarray((acc / l.reshape(B * n_q, rows, 1)).to_numpy(),
                                   np.float32).reshape(B, n_q, rows, d_k))
            lses.append(np.asarray(lse.to_numpy(), np.float32).reshape(B, n_q, rows))
        out = np.concatenate(outs, axis=2)
        lse_all = np.concatenate(lses, axis=2)
        o_t = T.tensor(out.transpose(0, 2, 1, 3).reshape(B, Tlen, n_q * d_k))
        return np.asarray((o_t @ wo.transpose(0, 1)).to_numpy(), np.float32), lse_all
