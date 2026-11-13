# -*- coding: utf-8 -*-
"""交叉验证：ORV 池计数 vs nvidia-smi 实测（判断我们的显存数字是否可信）。"""
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402
from v3_orv_train import build  # noqa: E402


def sampler(stop, out):
    peak = 0
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout
            peak = max(peak, int(o.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.02)
    out["peak"] = peak


def layer(x, L, cos, sin, d_k, n_q, n_kv):
    h = x + V.attention_orv(V.rmsnorm_orv(x, L["ln1"]), L["wq"], L["wk"], L["wv"],
                            L["wo"], cos, sin, d_k, n_q, n_kv, flash=True)
    Tlen = int(h.shape[1]); d = int(h.shape[2])
    hn = V.rmsnorm_orv(h, L["ln2"]).reshape(Tlen, d)
    eids, g, _s = MOE.route_top1(hn, L["router"])
    mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
    return h + mo.reshape(1, Tlen, d)


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    seq = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    base = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.splitlines()[0])
    cfg = dict(d=512, d_ff=2048, E=8, layers=1, n_q=8, n_kv=2, vocab=4096,
               seq=seq, steps=1)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    L = P["layers"][0]
    d_k = 64
    cos, sin = V.rope_cos_sin(seq, d_k, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    ids = rng.integers(0, 4096, seq).astype(np.int64)

    stop = threading.Event(); prof = {}
    th = threading.Thread(target=sampler, args=(stop, prof), daemon=True); th.start()

    x = T.tensor(P["emb"][ids].to_numpy().reshape(1, seq, 512), requires_grad=True)
    y = layer(x, L, cos, sin, d_k, 8, 2)
    loss = T.functional.cross_entropy(
        V.rmsnorm_orv(y, P["ln_f"]).reshape(seq, 512) @ P["head"].transpose(0, 1),
        np.roll(ids, -1))
    loss.backward()

    time.sleep(0.2)
    stop.set(); th.join(timeout=3)
    st = rt.pool_stats()
    print("seq=%d  ORV: live=%.0f peak_live=%.0f peak=%.0f MiB"
          % (seq, st["live_bytes"] / 2**20, st["peak_live_bytes"] / 2**20,
             st["peak_bytes"] / 2**20))
    print("      nvidia-smi: 空载 %d MiB，峰值 %d MiB，净增 %d MiB"
          % (base, prof.get("peak", 0), prof.get("peak", 0) - base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
