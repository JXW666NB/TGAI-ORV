# -*- coding: utf-8 -*-
"""按阶段拆一个 Transformer block 的耗时，看 ORV 的时间花在哪。

## 为什么先做这个

"fp16 只给 1.11×，torch 给 2.17×" 这个差距有两种可能：
  (a) fp16 在 GEMM 上没打满（cast 开销 / 累加精度）；
  (b) GEMM 只占整步的一部分，其余（注意力、softmax、norm、elemwise）仍是 fp32，
      所以整步加速比天然被摊薄。
本脚本用 `--fp16` 跑两组独立进程，逐阶段对比，直接区分 (a) 和 (b)。

用法（TG_FP16 必须在 import 前设好，所以分两次进程跑）：
    python profile_block.py            # fp32
    python profile_block.py --fp16     # 开 TG_FP16=1
"""
import io
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FP16 = "--fp16" in os.sys.argv
if FP16:
    os.environ["TG_FP16"] = "1"

import tgai_go.train as T                       # noqa: E402
from tgai_go.train import functional as Fn      # noqa: E402

OUT = os.path.join(HERE, "profile_block.txt")

B, S, D, NH, FF = 8, 2048, 1280, 10, 5120
DH = D // NH


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)

    def dev(n, m):
        return T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.02, (n, m)).astype(np.float32)))

    x = dev(B * S, D)
    qkv_w = dev(3 * D, D)
    o_w = dev(D, D)
    g_w = dev(FF, D)
    u_w = dev(FF, D)
    d_w = dev(D, FF)

    timings = {}

    def timeit(name, fn, reps=5):
        # 用 perf_counter + sync：各阶段都在毫秒级以上，Windows 上分辨率足够。
        # （ORV 没有把 cuEventElapsedTime 封装成公开方法，所以不用 CUDA event。）
        for _ in range(2):                     # 预热
            fn()
        rt.sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        rt.sync()
        ms = (time.perf_counter() - t0) * 1e3 / reps
        timings[name] = timings.get(name, 0.0) + ms

    # 逐阶段：只测前向（反向量级相同，比例一致）
    timeit("qkv_gemm", lambda: x @ qkv_w.T)
    z = x @ qkv_w.T

    # ★ 把 attention 拆开：闭包里的 permute/reshape 在 CUDA 上是**真拷贝**，
    #   之前把 33 ms 全算作"注意力"，可能大部分是我自己的布局搬运。
    def qkv_split():
        q, k, v = z.reshape(B, S, 3, NH, DH).permute(2, 0, 3, 1, 4)
        return (q.reshape(B * NH, S, DH), k.reshape(B * NH, S, DH),
                v.reshape(B * NH, S, DH))
    timeit("attn_layout_split", qkv_split)

    qq, kk, vv = qkv_split()
    timeit("attn_sdpa_only",
           lambda: Fn.scaled_dot_product_attention(qq, kk, vv, is_causal=True))

    y = Fn.scaled_dot_product_attention(qq, kk, vv, is_causal=True)

    def back_layout():
        return y.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
    timeit("attn_layout_back", back_layout)

    yh = back_layout()
    timeit("out_gemm", lambda: yh @ o_w.T)
    timeit("gate_gemm", lambda: x @ g_w.T)
    timeit("up_gemm", lambda: x @ u_w.T)

    gg = x @ g_w.T
    uu = x @ u_w.T
    timeit("silu_mul", lambda: Fn.silu(gg) * uu)

    h = Fn.silu(gg) * uu
    timeit("down_gemm", lambda: h @ d_w.T)

    total = sum(timings.values())
    rep = ["配置 B=%d S=%d D=%d NH=%d dh=%d FF=%d  模式=%s"
           % (B, S, D, NH, DH, FF, "fp16" if FP16 else "fp32"),
           "",
           "| 阶段 | ms | 占比 |",
           "| --- | --- | --- |"]
    for k in sorted(timings, key=lambda n: -timings[n]):
        rep.append("| %s | %.2f | %.1f%% |" % (k, timings[k], 100.0 * timings[k] / total))
    rep.append("| **合计** | **%.2f** | 100%% |" % total)

    gemm = sum(v for k, v in timings.items() if k.endswith("_gemm"))
    rep.append("")
    rep.append("GEMM 占比：**%.1f%%**（GEMM %.2f ms / 其余 %.2f ms）"
               % (100.0 * gemm / total, gemm, total - gemm))
    rep.append("")
    rep.append("判读：GEMM 占比越高，做 fp16 GEMM 的整步收益上限越大；")
    rep.append("若 GEMM 只占一半，整步加速比的天花板就是 1/(1-p+p/2)。")

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
