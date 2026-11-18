# -*- coding: utf-8 -*-
"""V3 模型的参数构成：专家 vs 非专家，各占多少。

为什么要这个数：用户提出"按专家切片做 EP（专家并行）"。
EP 能省多少，完全取决于专家权重在总参数里的占比 —— 不猜，直接按 V3 真实配置算。
做法：只建 **1 层**（各层结构相同），按名字归类求和，再乘层数。
"""
import collections
import re

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cpu")

CFG = dict(d=1024, d_ff=3328, E=4, layers=1, n_q=16, n_kv=2, vocab=96000,
           seq=512, attn_res=False, attn_res_block=2, top_k=2, csa=4,
           n_kv_attn=16, rope_theta=500000.0, csa_compress=4,
           csa_conv_kernel=3, csa_local_window=0)
LAYERS = 24
PAD_VOCAB = 98304          # 真实训练里词表会补齐到 98304


def bucket(name):
    if re.search(r"W[123](\.|$)", name) or "expert" in name.lower():
        return "专家 / experts"
    if "shared" in name.lower():
        return "共享专家 / shared"
    if name.startswith("head") or "lm_head" in name:
        return "输出层 / head"
    if name.startswith("emb") or "embed" in name.lower():
        return "嵌入 / embedding"
    return "其余（注意力/归一化）"


def main():
    cfg = dict(CFG)
    P = CLI.build(cfg, np.random.default_rng(0))
    params = CLI.params_of(P)
    tot = collections.Counter()
    cnt = collections.Counter()
    for name, p in zip(CLI.param_names(P) if hasattr(CLI, "param_names") else [], []):
        pass
    # params_of 只给张量；名字要通过 P 的结构遍历（与 ckpt 的 itern_named_tensors 同源）
    named = []
    try:
        from tgai_go.train import ckpt as _ck
        st = {}
        for t in params:
            pass
    except Exception:
        pass
    # 直接按 P 的嵌套 dict 递归取名字
    def walk(obj, prefix):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, prefix + "." + str(k) if prefix else str(k))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, "%s.%d" % (prefix, i))
        elif hasattr(obj, "_gpu") or hasattr(obj, "shape"):
            try:
                n = int(np.prod([int(s) for s in obj.shape]))
            except Exception:
                return
            named.append((prefix, n))

    walk(P, "")
    for nm, n in named:
        b = bucket(nm)
        tot[b] += n
        cnt[b] += 1
    allp = sum(tot.values())
    print("=== V3 单层参数构成（d=1024 d_ff=3328 E=4 top_k=2 csa=4）===")
    print("  %-22s %12s %8s %10s" % ("类别", "参数", "张量数", "占比"))
    for k, v in tot.most_common():
        print("  %-22s %12d %8d %9.2f%%" % (k, v, cnt[k], 100.0 * v / allp))
    print("  %-22s %12d %8d %9.2f%%" % ("单层合计", allp, sum(cnt.values()), 100.0))
    big = allp * LAYERS
    # ★ 修正：嵌入层与输出层**只出现一次**，不属于"每层"，不能乘层数。
    once = tot["嵌入 / embedding"] + tot["输出层 / head"]
    big = allp * LAYERS - once * (LAYERS - 1)
    print("\n  按 %d 层外推：%.3f B 参数 / %.3f GiB(fp32)"
          % (LAYERS, big / 1e9, big * 4 / float(1 << 30)))
    print("  （嵌入+输出层 %.3f B 只算一次）" % (once / 1e9))
    exp = tot["专家 / experts"] * LAYERS
    rest = big - exp
    print("  其中**专家**：%.3f B（%.1f%%）；非专家：%.3f B（%.1f%%）"
          % (exp / 1e9, 100.0 * exp / big, rest / 1e9, 100.0 * rest / big))
    for w in (2, 4, 8):
        print("    %d 卡 EP：每卡专家 %.3f B + 非专家全量 %.3f B = %.3f B（原 %.3f B）"
              % (w, exp / w / 1e9, rest / 1e9, (exp / w + rest) / 1e9, big / 1e9))


if __name__ == "__main__":
    main()
