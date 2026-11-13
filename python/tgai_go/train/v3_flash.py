# -*- coding: utf-8 -*-
"""分块在线 softmax 注意力（前向）：照 PyTorch mem-efficient 的算法，不物化 S×S。

算法（对齐 `kernel_forward.h`）：

```
对每个 query 块（Cq 行）：
    m = -inf, l = 0, O = 0
    对每个 key 块（Ck 行，因果时只到本块末尾）：
        S_b = Q_b·K_bᵀ·scale
        m_new = max(m, rowmax(S_b))
        l     = l·exp(m - m_new) + Σexp(S_b - m_new)
        O     = O·exp(m - m_new) + softmax(S_b - m_new)·V_b
        m     = m_new
    保存 O 与 LSE = m + log(l)
```

当前只做**前向**（`no_grad` 下跑，只留 O 与 LSE）。反向要重算，是下一步。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))


def _blk(t, i, c, rows):
    n = int(rows)
    hi = min((i + 1) * c, n)
    return t[:, :, i * c:hi, :]


def attention_flash_forward(x, wq, wk, wv, wo, cos, sin, rot_dim, n_q, n_kv,
                            cq=256, ck=256, causal=True):
    """返回 (out_numpy, lse_numpy)。x 形状 (B,T,d)；权重为 ORV 张量。"""
    import tgai_go.train as T
    from tgai_go.train import v3_ops as V

    B = int(x.shape[0]); Tlen = int(x.shape[1]); n_q = int(n_q); n_kv = int(n_kv)
    d_k = int(wq.shape[0]) // n_q

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

        scale = 1.0 / np.sqrt(float(d_k))
        outs, lses = [], []
        for qi in range(0, Tlen, cq):
            qb = q[:, :, qi:qi + cq, :]
            rows = int(qb.shape[2])
            m = np.full((B, n_q, rows), -np.inf, np.float32)
            l = np.zeros((B, n_q, rows), np.float32)
            acc = np.zeros((B, n_q, rows, d_k), np.float32)
            # 因果：key 只到本块末尾
            kmax = min(Tlen, qi + rows) if causal else Tlen
            for ki in range(0, kmax, ck):
                kb = k[:, :, ki:min(ki + ck, kmax), :]
                vb = v[:, :, ki:min(ki + ck, kmax), :]
                sb = (qb @ kb.transpose(0, 1).transpose(2, 3)) * scale \
                    if False else ((qb.reshape(B * n_q, rows, d_k)
                                    @ kb.reshape(B * n_q, int(kb.shape[2]), d_k).transpose(1, 2)) * scale)
                sb = np.asarray(sb.to_numpy(), np.float32).reshape(B, n_q, rows, -1)
                cols = sb.shape[-1]
                if causal:
                    abs_q = np.arange(rows)[:, None] + qi
                    abs_k = np.arange(cols)[None, :] + ki
                    sb = np.where(abs_k <= abs_q, sb, -np.inf)
                m_new = np.maximum(m, sb.max(-1))
                m_new = np.where(np.isfinite(m_new), m_new, 0.0)
                e = np.exp(sb - m_new[..., None])
                e = np.where(np.isfinite(sb), e, 0.0)
                l = l * np.exp(m - m_new) + e.sum(-1)
                acc = acc * np.exp(m - m_new)[..., None] + (e @ np.asarray(vb.to_numpy(), np.float32).reshape(B, n_q, cols, d_k))
                m = m_new
            lse = m + np.log(np.maximum(l, 1e-20))
            outs.append((acc / np.maximum(l, 1e-20)[..., None]).reshape(B, n_q, rows, d_k))
            lses.append(lse)
        out = np.concatenate(outs, axis=2)
        lse_all = np.concatenate(lses, axis=2)
        o_t = T.tensor(out.transpose(0, 2, 1, 3).reshape(B, Tlen, n_q * d_k))
        y = o_t @ wo.transpose(0, 1)
        return np.asarray(y.to_numpy(), np.float32), lse_all
