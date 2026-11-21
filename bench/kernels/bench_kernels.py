# -*- coding: utf-8 -*-
"""逐内核基准：ORV vs torch vs 硬件上限。

为什么需要它：端到端步时会被数据加载 / Python 开销 / 分配器噪声掩盖，
改一个内核 ±5% 根本看不出来，也就无法真的改进内核质量。
这个脚本把"内核质量"变成可测量的数字：**达到峰值 FLOPS 的百分比**或**带宽百分比**。

判据：任何内核改动必须让对应行百分比上升，且端到端不掉。

用法
    python bench_kernels.py --probe                 # 打印本机峰值/带宽表与 cc
    python bench_kernels.py --only orv              # 只测 ORV
    python bench_kernels.py --only torch            # 只测 torch
    python bench_kernels.py --shapes 2048,4096      # 指定序列长度
"""
import argparse
import json
import statistics
import sys
import time

import numpy as np

# 峰值表：fp32 TFLOPS / 显存带宽 GB/s / 动态 smem KB（按 cc 粗分，够用即可）
PEAKS = {
    (6, 0): dict(name="Tesla P100", fp32=9.3, bw=732, smem=48),
    (7, 0): dict(name="Tesla V100", fp32=14.0, bw=900, smem=96),
    (7, 5): dict(name="Tesla T4", fp32=8.1, bw=320, smem=64),
    (8, 6): dict(name="A10 / 3090", fp32=31.2, bw=600, smem=100),
    (8, 9): dict(name="RTX 4090", fp32=82.6, bw=1008, smem=100),
    (9, 0): dict(name="A100/H100 类", fp32=19.5, bw=1555, smem=164),
    (12, 0): dict(name="RTX 5090", fp32=104.8, bw=1792, smem=228),
}


def peak_of(cc):
    if cc in PEAKS:
        return PEAKS[cc]
    # 退到同 major 的最近一档
    cand = [k for k in PEAKS if k[0] == cc[0]]
    return PEAKS[cand[0]] if cand else dict(name="unknown", fp32=0.0, bw=0.0, smem=48)


def timeit(fn, warmup=3, iters=10):
    """返回中位数秒。fn 里不要自己 sync（这里统一切）。"""
    for _ in range(warmup):
        fn()
    import torch
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# ------------------------------------------------------------------ 形状定义
def cases(seq_list):
    out = []
    for S in seq_list:
        out.append(dict(name="attn_fwd", S=S, B=2, H=18, Dh=64, kind="flops",
                        flops=lambda S=S: 4 * 2 * 18 * S * S * 64))
        out.append(dict(name="attn_bwd", S=S, B=2, H=18, Dh=64, kind="flops",
                        flops=lambda S=S: 2.5 * 4 * 2 * 18 * S * S * 64))
    N = 2 * 2048 * 1152
    out.append(dict(name="softmax", N=N, kind="bw", bytes=lambda N=N: 2 * N * 4))
    out.append(dict(name="rmsnorm", N=N, kind="bw", bytes=lambda N=N: 2 * N * 4))
    out.append(dict(name="silu", N=N, kind="bw", bytes=lambda N=N: 2 * N * 4))
    out.append(dict(name="gemm", M=4096, K=1152, Nn=1152, kind="flops",
                    flops=lambda: 2 * 4096 * 1152 * 1152))
    out.append(dict(name="moe_gather", tokens=4096, topk=5, D=1152, kind="bw",
                    bytes=lambda: 2 * 4096 * 5 * 1152 * 4))
    return out


# --------------------------------------------------------------------- ORV 侧
def bench_orv(cases_, only_names=None):
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    T.set_default_device("cuda")
    rt = T._cuda()
    caps = rt.device_caps()
    print("\n== ORV ==")
    rows = []
    rng = np.random.default_rng(0)

    def dev(shape):
        return T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.02, shape).astype(np.float32)))

    for c in cases_:
        if only_names and c["name"] not in only_names:
            continue
        name = c["name"]
        B, H, Dh = c.get("B", 2), c.get("H", 18), c.get("Dh", 64)
        S = c.get("S")
        try:
            if name.startswith("attn_"):
                q = dev((B * H, S, Dh))
                k = dev((B * H, S, Dh))
                v = dev((B * H, S, Dh))
                for t in (q, k, v):
                    t.requires_grad = True

                def run_fwd():
                    return Fn.scaled_dot_product_attention(q, k, v, is_causal=True)

                if name == "attn_fwd":
                    dt = timeit(run_fwd)
                else:
                    def run_bwd():
                        o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
                        o.backward(T.ones(tuple(o.shape)))
                    dt = timeit(run_bwd)
            elif name == "softmax":
                x = dev((2048, 1152))

                def run():
                    return Fn.softmax(x, axis=-1)
                dt = timeit(run)
            elif name == "rmsnorm":
                x = dev((2048, 1152))
                _norm = T.nn.RMSNorm(1152)

                def run():
                    return _norm(x)
                dt = timeit(run)
            elif name == "silu":
                x = dev((2048, 1152))

                def run():
                    return Fn.silu(x)
                dt = timeit(run)
            elif name == "gemm":
                a = dev((c["M"], c["K"]))
                b = dev((c["K"], c["Nn"]))

                def run():
                    return a @ b
                dt = timeit(run)
            elif name == "moe_gather":
                x = dev((c["tokens"], c["D"]))
                idx = T.tensor(rng.integers(0, c["tokens"],
                                            (c["tokens"] * c["topk"],)).astype(np.int64))

                def run():
                    return x[idx]          # 行 gather（Tensor.__getitem__）
                dt = timeit(run)
            else:
                continue
        except Exception as e:                      # noqa: BLE001
            print("  %-10s FAILED: %s" % (name, str(e)[:90]))
            continue
        val = c["flops"]() if c["kind"] == "flops" else c["bytes"]()
        achieved = val / dt / 1e12 if c["kind"] == "flops" else val / dt / 1e9
        rows.append(dict(impl="orv", name=name, S=S, seconds=dt,
                         value=achieved, kind=c["kind"]))
        print("  %-10s S=%-5s %8.4f s  %6.2f %s"
              % (name, S, dt, achieved, "TFLOPS" if c["kind"] == "flops" else "GB/s"))
    return rows, caps


# ------------------------------------------------------------------- torch 侧
def bench_torch(cases_, only_names=None):
    import torch
    import torch.nn.functional as F

    print("\n== torch ==")
    rows = []
    for c in cases_:
        if only_names and c["name"] not in only_names:
            continue
        name = c["name"]
        B, H, Dh = c.get("B", 2), c.get("H", 18), c.get("Dh", 64)
        S = c.get("S")
        try:
            if name.startswith("attn_"):
                q = torch.randn(B * H, S, Dh, device="cuda", requires_grad=True)
                k = torch.randn(B * H, S, Dh, device="cuda", requires_grad=True)
                v = torch.randn(B * H, S, Dh, device="cuda", requires_grad=True)
                if name == "attn_fwd":
                    dt = timeit(lambda: F.scaled_dot_product_attention(
                        q, k, v, is_causal=True))
                else:
                    def run_bwd():
                        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                        o.backward(torch.ones_like(o))
                        q.grad = k.grad = v.grad = None
                    dt = timeit(run_bwd)
            elif name == "softmax":
                x = torch.randn(2048, 1152, device="cuda")
                dt = timeit(lambda: F.softmax(x, dim=-1))
            elif name == "rmsnorm":
                x = torch.randn(2048, 1152, device="cuda")
                w = torch.randn(1152, device="cuda")
                dt = timeit(lambda: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)
                                                    + 1e-6) * w)
            elif name == "silu":
                x = torch.randn(2048, 1152, device="cuda")
                dt = timeit(lambda: F.silu(x))
            elif name == "gemm":
                a = torch.randn(c["M"], c["K"], device="cuda")
                b = torch.randn(c["K"], c["Nn"], device="cuda")
                dt = timeit(lambda: a @ b)
            elif name == "moe_gather":
                x = torch.randn(c["tokens"], c["D"], device="cuda")
                idx = torch.randint(0, c["tokens"], (c["tokens"] * c["topk"],),
                                    device="cuda")
                dt = timeit(lambda: x.index_select(0, idx))
            else:
                continue
        except Exception as e:                      # noqa: BLE001
            print("  %-10s FAILED: %s" % (name, str(e)[:90]))
            continue
        val = c["flops"]() if c["kind"] == "flops" else c["bytes"]()
        achieved = val / dt / 1e12 if c["kind"] == "flops" else val / dt / 1e9
        rows.append(dict(impl="torch", name=name, S=S, seconds=dt,
                         value=achieved, kind=c["kind"]))
        print("  %-10s S=%-5s %8.4f s  %6.2f %s"
              % (name, S, dt, achieved, "TFLOPS" if c["kind"] == "flops" else "GB/s"))
    return rows


def probe():
    import torch
    if not torch.cuda.is_available():
        print("torch 无 CUDA，无法测；修法：在有 N 卡的机器上跑。")
        return 2
    cc = torch.cuda.get_device_capability(0)
    pk = peak_of(cc)
    print("设备: %s  cc=%d.%d" % (torch.cuda.get_device_name(0), cc[0], cc[1]))
    print("表中峰值: %s  fp32 %.1f TFLOPS 带宽 %d GB/s 动态 smem %d KB"
          % (pk["name"], pk["fp32"], pk["bw"], pk["smem"]))
    print("\n验收线（见 docs/KERNEL_QUALITY_STRATEGY.md §4）:")
    print("  attn_fwd >= 45%% peak | attn_bwd >= 35%% | softmax/rmsnorm >= 85%% bw"
          " | gemm >= 70%% | moe_gather >= 80%% bw")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="逐内核基准：ORV vs torch vs 硬件上限")
    ap.add_argument("--only", choices=("orv", "torch"))
    ap.add_argument("--shapes", default="2048", help="逗号分隔序列长度")
    ap.add_argument("--kernels", default="", help="逗号分隔内核名过滤，如 attn_fwd,gemm")
    ap.add_argument("--json", default=None)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args(argv)

    if a.probe:
        return probe() or 0

    seqs = [int(x) for x in a.shapes.split(",")]
    cases_ = cases(seqs)
    names = {x.strip() for x in a.kernels.split(",") if x.strip()} or None

    all_rows = []
    cc = None
    if a.only in (None, "orv"):
        rows, caps = bench_orv(cases_, names)
        all_rows += rows
        cc = caps.get("cc")
    if a.only in (None, "torch"):
        all_rows += bench_torch(cases_, names)
        if cc is None:
            import torch
            cc = torch.cuda.get_device_capability(0)

    pk = peak_of(tuple(cc) if cc else (0, 0))
    print("\n== 对照硬件上限（%s）==" % pk["name"])
    for r in all_rows:
        cap = pk["fp32"] if r["kind"] == "flops" else pk["bw"]
        pct = (r["value"] / cap * 100.0) if cap else 0.0
        print("  %-6s %-10s S=%-5s %6.2f %-6s = 峰值 %.0f%%"
              % (r["impl"], r["name"], r["S"], r["value"],
                 "TFLOPS" if r["kind"] == "flops" else "GB/s", pct))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(device=pk["name"], cc=list(cc) if cc else None,
                           rows=all_rows), fh, ensure_ascii=False, indent=2)
        print("\n结果写入 %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
