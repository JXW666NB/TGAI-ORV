# -*- coding: utf-8 -*-
"""embedding 设备化的**可信计时**：用 torch.cuda.synchronize()（整设备）而不是只同步一条流。

为什么要重做：上一版用 rt.sync() 得到 "0.0 ms/步"，但 emb_bwd 每次都要 zeros([V,D])
（V2.7 下 134 MB memset）+ 写 seq 行，物理上不可能 0 ms —— 说明计时没覆盖设备工作
（多流或流同步语义）。这里：
  1. 先用**已知代价**的对照（20 次 134 MB memset）验证 harness 本身；
  2. 再分别计时旧的 numpy 路径与新的设备路径。

运行（需独占 GPU）：python embed_timing2.py
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
SEQ = 128
REP = 20


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    def sync():
        torch.cuda.synchronize()          # 整设备同步，不受流语义影响

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    cfg = dict(d_model=int(h["d_model"]), n_heads=int(h["n_heads"]), d_ff=int(h["d_ff"]),
               n_experts=int(h["n_experts"]), n_activated=int(h["n_activated"]),
               n_layers=int(h["n_layers"]), rope_theta=float(h["rope_theta"]))
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    tr.attach_head(st)
    rt = T._cuda()
    w = tr.head.weight
    V, D = int(w.shape[0]), int(w.shape[1])
    log("V=%d D=%d  (一张 [V,D] = %.1f MB)" % (V, D, V * D * 4 / 1024 ** 2))

    rng = np.random.default_rng(0)
    ids = rng.integers(0, V, SEQ, dtype=np.int64)
    g = rng.normal(0, 1, (SEQ, D)).astype(np.float32)
    w_np = np.asarray(w.to_numpy(), np.float32)
    g_buf = rt.h2d(np.ascontiguousarray(g))

    # ---- 对照：已知代价（20 次 134 MB memset）----
    rt.zeros((V, D)); sync()
    t0 = time.time()
    for _ in range(REP):
        rt.zeros((V, D))
    sync()
    ctrl = (time.time() - t0) / REP
    log("对照 memset(%.0f MB)：%.3f ms/次 ⇒ 有效带宽 %.0f GB/s"
        % (V * D * 4 / 1024 ** 2, ctrl * 1000, (V * D * 4) / max(ctrl, 1e-9) / 1e9))

    # ---- 旧路径 ----
    def old_once():
        xb = rt.h2d(np.ascontiguousarray(w_np[ids]))
        tr.head.weight.grad = None
        tr.add_embed_grad(ids, g)

    old_once(); sync()
    t0 = time.time()
    for _ in range(REP):
        old_once()
    sync()
    t_old = (time.time() - t0) / REP

    # ---- 设备路径 ----
    def new_once():
        xb, ib = tr.x0_from_ids(ids)
        tr.head.weight.grad = None
        tr.add_embed_grad_device(ib, g_buf)

    new_once(); sync()
    t0 = time.time()
    for _ in range(REP):
        new_once()
    sync()
    t_new = (time.time() - t0) / REP

    log("旧路径（numpy+h2d / D2H 相加）: %.2f ms/次" % (t_old * 1000))
    log("设备路径（gather_rows / emb_bwd+add）: %.2f ms/次" % (t_new * 1000))
    log("省 %.2f ms（%.1fx）；按 %d 步/秒算，每步省 %.0f ms"
        % ((t_old - t_new) * 1000, t_old / max(t_new, 1e-9), 1, (t_old - t_new) * 1000))
    st.close()
    rt.empty_cache()


if __name__ == "__main__":
    main()
