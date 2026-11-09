# -*- coding: utf-8 -*-
"""诊断：V27 单步耗时里 **kernel 发射开销** vs **算力** 各占多少。

只做前向+反向（不建优化器，避免多占 11 GB），这样才能在 S=512 下跑得动。

  A. 固定 seq=128，层数 2/4/8/16/28：每层耗时是否恒定（≈发射受限）还是随规模变（算力受限）
  B. 固定 8 层，seq=128/256/512：每层耗时随 seq 怎么变

运行（需独占 GPU）：
    python speed_profile.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
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
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    tr.attach_head(st)
    emb = st["token_embedding.weight"]
    log("显存（仅权重）: %d MiB" % RUN.vram())

    def one_step(nlayer, seq):
        ids = np.random.default_rng(0).integers(0, 30000, seq + 1, dtype=np.int64)
        j = np.arange(0, DH, 2, dtype=np.float32) / DH
        ang = np.repeat(np.arange(seq, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
        cos = T.tensor(np.cos(ang).astype(np.float32)); sin = T.tensor(np.sin(ang).astype(np.float32))
        x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)

        def run():
            x = T.tensor(x0)
            for i in range(nlayer):
                x = tr.fwd(x, i, cos, sin)
            lg = tr.head(tr.final_norm(x))
            ls = T.functional.cross_entropy(lg, ids[1:])
            ls.backward()
            for i in range(nlayer):
                for p in tr.params_of(i):
                    p.grad = None
            for p in tr.head_params():
                p.grad = None
        run()                               # 预热：NVRTC 编译 + cuBLAS 工作区
        T._cuda().sync()
        t0 = time.time()
        for _ in range(3):
            run()
        T._cuda().sync()
        dt = (time.time() - t0) / 3.0
        T._cuda().empty_cache()
        return dt

    log("=== A. 层数扫描（seq=128，前向+反向，无优化器）===")
    prev = None
    for n in (2, 4, 8, 16, 28):
        dt = one_step(n, 128)
        inc = "" if prev is None else "   增量 %.1f ms/层" % ((dt - prev[1]) / (n - prev[0]) * 1000)
        log("  层数 %-3d  单步 %.3f s   平均 %.1f ms/层%s" % (n, dt, dt / n * 1000, inc))
        prev = (n, dt)

    log("=== B. 序列扫描（8 层）===")
    for seq in (128, 256, 512):
        dt = one_step(8, seq)
        log("  seq=%-4d 单步 %.3f s   %.1f ms/层   %.3f ms/千token"
            % (seq, dt, dt / 8 * 1000, dt / 8 / seq * 1000 * 1000))

    st.close()
    T._cuda().empty_cache()


if __name__ == "__main__":
    main()
