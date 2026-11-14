# -*- coding: utf-8 -*-
"""三方对比注意力：物化 / 块化(flash) / 原生内核(lowmem)。d_k 可配。

lowmem 走引擎自带的 tgai_attn_nop.cu，但 `_sdpa_lowmem_ok` 要求 dh==64，
所以 d_k=128 时会静默回退到物化 —— 用这个探针就能看出来（差值恒为 0 即回退）。
"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    d, S = 512, 1024
    n_q = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_kv, d_k = 2, d // (int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    W = lambda *s: T.tensor(rng.normal(0, .02, s).astype(np.float32), requires_grad=True)
    ln = W(d)
    wq = W(n_q * d_k, d)
    wk = W(n_kv * d_k, d)
    wv = W(n_kv * d_k, d)
    wo = W(d, n_q * d_k)
    cos, sin = V.rope_cos_sin(S, d_k, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    x0 = rng.normal(0, .3, (1, S, d)).astype(np.float32)
    _ok = T._sdpa_lowmem_ok(
        T.tensor(np.zeros((1, S, d_k), np.float32)),
        T.tensor(np.zeros((1, S, d_k), np.float32)),
        T.tensor(np.zeros((1, S, d_k), np.float32)),
        True, d_k)
    print("n_q=%d d_k=%d  lowmem 可用=%s" % (n_q, d_k, _ok))

    ref = ref_dx = None
    # ★ 预热：lowmem 走 NVRTC 编译的 tgai_attn_nop 内核，首次调用含编译时间（实测约 3.7 s）
    rt.ensure_attn_nop()
    for kw in (dict(), dict(flash=True), dict(lowmem=True)):
        xw = T.tensor(x0.copy(), requires_grad=True)
        yw = V.attention_orv(V.rmsnorm_orv(xw, ln), wq, wk, wv, wo, cos, sin, d_k,
                             n_q, n_kv, **kw)
        yw.to_numpy()
        del xw, yw
    rt.empty_cache()
    for mode, kw in (("materialized", {}), ("flash", dict(flash=True)),
                     ("lowmem", dict(lowmem=True))):
        rt.empty_cache()
        p0 = rt.pool_stats()["peak_live_bytes"]
        x = T.tensor(x0.copy(), requires_grad=True)
        t0 = time.time()
        y = V.attention_orv(V.rmsnorm_orv(x, ln), wq, wk, wv, wo, cos, sin, d_k,
                            n_q, n_kv, **kw)
        o = np.asarray(y.to_numpy(), np.float32)
        y.sum().backward()
        dt = time.time() - t0
        pk = rt.pool_stats()["peak_live_bytes"]
        dx = np.asarray(x.grad.to_numpy(), np.float32)
        if ref is None:
            ref, ref_dx = o, dx
            print("%-12s %7.3fs  峰值 %6.0f MiB  （基准）" % (mode, dt, (pk - p0) / 2**20))
        else:
            print("%-12s %7.3fs  峰值 %6.0f MiB  输出差 %.2e  dx 差 %.2e"
                  % (mode, dt, (pk - p0) / 2**20,
                     float(np.abs(o - ref).max()), float(np.abs(dx - ref_dx).max())))
        del x, y
        rt.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
