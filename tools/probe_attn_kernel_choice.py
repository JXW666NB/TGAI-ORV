# -*- coding: utf-8 -*-
"""报告本次运行实际启动的注意力内核名，以及本卡动态 smem 上限。

用途：证明 V100（96 KB）仍走常驻版 `attn_bwd_flash`，
而模拟 48 KB 时自动切到窗口版 `attn_bwd_flash_w`。

用法：
    python tools/probe_attn_kernel_choice.py            # 真实上限
    python tools/probe_attn_kernel_choice.py --limit-kb 48
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-kb", type=int, default=None)
    args = ap.parse_args()

    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    T.set_default_device("cuda")
    rt = T._cuda()
    if args.limit_kb:
        rt._device_smem_limit = lambda: args.limit_kb * 1024
    used = []
    orig = rt.launch

    def spy(name, *a, **k):
        used.append(name)
        return orig(name, *a, **k)

    rt.launch = spy

    bh, seq, dh = 2, 64, 64
    rng = np.random.default_rng(0)
    ts = []
    for _ in range(3):
        a = rng.normal(0, .3, (bh, seq, dh)).astype(np.float32)
        ts.append(T.Tensor._from_gpu(rt.h2d(a), requires_grad=True))
    q, k, v = ts
    o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)
    o.backward(T.Tensor._from_gpu(rt.h2d(np.ones((bh, seq, dh), np.float32))))

    lim = rt._device_smem_limit() or 0
    attn = [n for n in used if "attn" in n]
    print("LIMIT_KB %d" % (lim // 1024))
    print("LAUNCHED %s" % sorted(set(attn)))
    bwd = [n for n in attn if "bwd" in n]
    print("BACKWARD %s" % (bwd[0] if bwd else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
