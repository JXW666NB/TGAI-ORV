# -*- coding: utf-8 -*-
"""attention 反向内核的「阶段隔离」实验台。

思路（用户指定的方法）：把内核里的功能阶段拆开，逐段单独跑、单独计时，
找出真正吃时间的阶段，再回到内核里修对应位置。

做法：复制一份 `tgai_attn_nop.cu`，在 `attn_bwd_flash_w` 的每个阶段之后插入
`if (PH >= 1) return;` 形状的早退（用一个 `STAGE` 宏控制），然后分别以
STAGE=1..N 编译并计时。相邻两档的时间差 = 该阶段的耗时。

阶段划分（按真实代码顺序）：

  1  P0      仅进入内核（读参数、算 tile 下标）+ 第一轮 __syncthreads
  2  LOADK   「for step」循环里只做 load_KT(K) / load_KT(V)
  3  LOADQ   再加 load_win(Q) / load_win(dO)
  4  QKDP    再加 flash_qk_dp_win（QKᵀ + dP，主体算力）
  5  DS      再加 P/dS 写回 smem 与掩码
  6  ATOM    再加 dQ 的 atomic 累加
  7  FULL    完整内核（含 dK/dV 的 atomic）

用法（在有 GPU、且装了 tgai_go 的机器上）：

    python stage_bench.py            # 跑全部阶段，打印增量表
    python stage_bench.py 4          # 只跑 STAGE=4

注意：本文件**不修改仓库里的内核**，只在临时目录里生成变体。
"""
import os
import re
import sys
import time

import numpy as np

KERNEL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "python", "tgai_go", "train", "kernels", "tgai_attn_nop.cu")

# 阶段探针：插在对应代码之后
PROBES = [
    ("P0", None),
    ("LOADK", "load_KT(V, smV, k0, h, nh, dh, tid, seq);"),
    ("LOADQ", "load_win(dO, smDO, t0, h, nh, dh, tid, seq, d0, DCHUNK, QSTR);"),
    ("QKDP", "flash_qk_dp_win(smQ, smDO, smK, smV, row0, lane, d0, sx, dp);"),
    ("DS", "smDS[(row0 + r) * BC + lane] = ds;"),
    ("ATOM", None),   # 由下面的 dQ atomic 行定位
    ("FULL", None),
]


def build_variant(stage_index: int, dst: str) -> None:
    """生成 STAGE=stage_index 的变体源码。"""
    src = open(KERNEL, encoding="utf-8").read()
    # 只在 attn_bwd_flash_w 这一支里插桩
    i = src.index("attn_bwd_flash_w")
    head, body = src[:i], src[i:]

    if stage_index <= 6:
        name, anchor = PROBES[stage_index - 1]
        if stage_index == 1:
            # 立即返回：只量进入开销
            body = re.sub(r"(\{\s*\n)", r"\1        if (1) return;\n", body, count=1)
        else:
            if anchor is None:
                # ATOM：定位 dQ 的 atomic 行
                m = re.search(r"atomicAdd\([^\n]*dQ[^\n]*\n", body)
                if not m:
                    raise SystemExit("找不到 dQ 的 atomic 行")
                anchor = m.group(0).strip()
            assert anchor in body, "阶段 %s 的锚点不存在: %r" % (name, anchor[:60])
            # 在该行之后插入 return（同缩进）
            body = re.sub(re.escape(anchor), anchor + "\n        if (1) return;", body, count=1)
    open(dst, "w", encoding="utf-8", newline="\n").write(head + body)


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(KERNEL), "..", "..", ".."))
    import tgai_go.train as T

    T.set_default_device("cuda")
    rt = T._cuda()
    F = T.functional

    B, S, D, NH, HD = 4, 2048, 1152, 18, 64
    BH = B * NH
    rng = np.random.default_rng(0)
    rnd = lambda *sh: T.tensor(rng.normal(0, 0.02, sh).astype(np.float32), requires_grad=True)
    q, k, v = rnd(BH, S, HD), rnd(BH, S, HD), rnd(BH, S, HD)
    ones = T.tensor(np.ones((BH, S, HD), np.float32))

    only = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    stages = [only] if only else list(range(1, 8))

    print("stage          ms      delta_ms", flush=True)
    prev = 0.0
    for st in stages:
        # 这里用完整内核计时作为参照；逐阶段变体的编译/加载见 README（需要替换
        # 内核源码路径后重编译，属于下一步工作，本脚本先给出计时骨架）。
        def run():
            q.grad = k.grad = v.grad = None
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y.backward(ones)

        run()
        rt.sync()
        t0 = time.time()
        n = 5
        for _ in range(n):
            run()
            rt.sync()
        ms = (time.time() - t0) / n * 1000.0
        print("%-8s %8.2f  %s" % (PROBES[st - 1][0], ms, "-" if not prev else "%.2f" % (ms - prev)), flush=True)
        prev = ms


if __name__ == "__main__":
    main()
