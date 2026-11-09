# -*- coding: utf-8 -*-
"""统计一层的前向/反向**发射了多少次 kernel**，并按内核名分类（找出小 kernel 大户）。

思路：给 rt.launch 装计数器（只统计名字），跑一层前向 + 反向，输出 Top-N。
运行（需独占 GPU）：python kernel_count.py
"""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
SEQ = 128
NL = 28


def main():
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    DM, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = DM // NH, float(h["rope_theta"])
    cfg = dict(d_model=DM, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    rt = T._cuda()

    counts = Counter()
    orig_launch = type(rt).launch

    def counted(self, name, *a, **kw):
        counts[name] += 1
        return orig_launch(self, name, *a, **kw)

    type(rt).launch = counted

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30000, SEQ, dtype=np.int64)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32))
    sin = T.tensor(np.sin(ang).astype(np.float32))
    x0 = np.ascontiguousarray(np.asarray(st["token_embedding.weight"][ids], np.float32))

    # 预热（NVRTC 编译不计入统计）
    y = tr.fwd(T.tensor(x0), 0, cos, sin)
    (y * T.tensor(np.ones((SEQ, DM), np.float32))).sum().backward()
    rt.sync()

    counts.clear()
    x = T.tensor(x0, requires_grad=True)
    y = tr.fwd(x, 0, cos, sin)
    rt.sync()
    fwd_counts = Counter(counts)
    fwd_total = sum(fwd_counts.values())

    counts.clear()
    g = np.ones((SEQ, DM), np.float32)
    (y * T.tensor(g)).sum().backward()
    rt.sync()
    bwd_counts = Counter(counts)
    bwd_total = sum(bwd_counts.values())

    log("一层前向发射 %d 次；反向发射 %d 次 ⇒ 一步（28 层）约 %d 次"
        % (fwd_total, bwd_total, (fwd_total + bwd_total) * NL))
    log("--- 前向 Top10 ---")
    for name, c in fwd_counts.most_common(10):
        log("   %-28s %d" % (name, c))
    log("--- 反向 Top10 ---")
    for name, c in bwd_counts.most_common(10):
        log("   %-28s %d" % (name, c))
    st.close()
    rt.empty_cache()


if __name__ == "__main__":
    main()
