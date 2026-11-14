# -*- coding: utf-8 -*-
"""给分配器插桩：看清一次 MoE 前向+反向到底分配了哪些块、由谁发起。

为什么需要它：
    静态读代码数不出分配点 —— `MoE.forward` 里只有 4 处 numpy 分配，真正的
    设备缓冲都在它调用的算子内部。实测 B=4 时 MoE 占 3.75 GB，而主缓冲只要约
    180 MB，差了 20 倍，必须先看是谁在重复分配。

归因规则（吃过一次亏）：
    `rt.zeros` -> `rt.full` -> `rt.empty`，真正调用 `empty` 的是 rt 自己的包装方法。
    所以栈回溯必须**跳过 cuda.py 里的所有帧**，取第一个业务侧调用者，
    否则每条都会记到 `cuda.py: zeros` 名下（第一次跑就踩了）。

用法：
    python tools/probe_alloc_sites.py [B] [top_k] [d_model] [d_ff]
"""
import collections
import sys
import traceback

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def _caller_key(frames):
    """栈里第一个不在 cuda.py 的帧（从最内层往外找）。"""
    for f in reversed(frames):
        if "cuda.py" not in (f.filename or ""):
            base = f.filename.replace("\\", "/").rsplit("/", 1)[-1]
            return "%s:%d %s" % (base, f.lineno, f.name)
    f = frames[-1]
    base = f.filename.replace("\\", "/").rsplit("/", 1)[-1]
    return "%s:%d %s" % (base, f.lineno, f.name)


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    D = int(sys.argv[3]) if len(sys.argv) > 3 else 1152
    D_FF = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    S, n_experts = 2048, 36

    T.set_default_device("cuda")
    rt = T._cuda()

    stats = collections.Counter()
    counts = collections.Counter()
    orig = rt.empty

    def instrumented_empty(*a, **kw):
        try:
            shape = a[0] if a else kw.get("shape", ())
            n = 1
            for d in shape:
                n *= int(d)
            nbytes = n * 4
            key = _caller_key(traceback.extract_stack(limit=10))
            stats[key] += nbytes
            counts[key] += 1
        except Exception:
            pass
        return orig(*a, **kw)

    rt.empty = instrumented_empty

    moe = T.nn.MoE(D, D_FF, n_experts, top_k=top_k, n_shared=1, shared_d_ff=576)
    for p in moe.parameters():
        p.requires_grad = True
    rng = np.random.default_rng(0)
    x = T.tensor(rng.normal(0, 0.02, (B * S, D)).astype(np.float32), requires_grad=True)

    y = moe(x)
    (y * y).sum().backward()          # 预热：内核编译不计入
    rt.sync()
    stats.clear()
    counts.clear()

    y = moe(x)
    (y * y).sum().backward()
    rt.sync()

    total = sum(stats.values())
    blocks = sum(counts.values())
    main_mb = B * S * top_k * D * 4 / 2**20
    print("B=%d top_k=%d D=%d d_ff=%d S=%d" % (B, top_k, D, D_FF, S), flush=True)
    print("一次前向+反向：分配 %d 块，合计 %.2f GB" % (blocks, total / 2**30), flush=True)
    print("主缓冲 (B*S*top_k, D) = %.1f MB" % main_mb, flush=True)
    print("%-40s %10s %8s" % ("发起位置", "合计MB", "块数"), flush=True)
    for key, nbytes in stats.most_common(16):
        print("%-40s %10.1f %8d" % (key, nbytes / 2**20, counts[key]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
