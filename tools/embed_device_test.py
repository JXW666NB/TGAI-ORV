# -*- coding: utf-8 -*-
"""embedding 路径设备化：`x0` 用 gather_rows 构造、梯度用 emb_bwd 设备内相加。

验证两件事：
  1. 与原来的 numpy 路径**逐位一致**（x0 缓冲、head.weight.grad）；
  2. 每步省掉 3 次 134 MB 级主机往返，实测省多少时间。

运行（需独占 GPU）：python embed_device_test.py
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


def main():
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

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
    tr.attach_head(st)          # head 只有在挂头之后才存在
    rt = T._cuda()
    w = tr.head.weight
    V, D = int(w.shape[0]), int(w.shape[1])
    log("V=%d D=%d" % (V, D))

    rng = np.random.default_rng(0)
    ids = rng.integers(0, V, SEQ, dtype=np.int64)
    w_np = np.asarray(w.to_numpy(), np.float32)

    # 1) x0 构造：numpy 索引+h2d  vs  gather_rows
    x_np = np.ascontiguousarray(w_np[ids])
    x_np_buf = rt.h2d(x_np)
    x_dev_buf, id_buf = tr.x0_from_ids(ids)
    a = np.asarray(rt.d2h(x_np_buf), np.float32)
    b = np.asarray(rt.d2h(x_dev_buf), np.float32)
    ok_x0 = np.array_equal(a.view(np.uint32), b.view(np.uint32))

    # 2) 输入 embedding 梯度：numpy 路径 vs 设备路径
    g = rng.normal(0, 1, (SEQ, D)).astype(np.float32)
    tr.head.weight.grad = None
    ok1 = tr.add_embed_grad(ids, g)
    old = np.asarray(tr.head.weight.grad.to_numpy(), np.float32).copy()
    old_buf = tr.head.weight.grad._gpu

    tr.head.weight.grad = None
    g_buf = rt.h2d(np.ascontiguousarray(g))
    ok2 = tr.add_embed_grad_device(id_buf, g_buf)
    new = np.asarray(tr.head.weight.grad.to_numpy(), np.float32).copy()
    ok_g = np.array_equal(old.view(np.uint32), new.view(np.uint32))

    log("x0 逐位一致=%s   梯度逐位一致=%s   (add_embed_grad=%s/%s)"
        % (ok_x0, ok_g, ok1, ok2))
    if not (ok_x0 and ok_g):
        d = np.abs(old - new).max()
        log("最大差 %.3e" % float(d))

    # 3) 计时（各 20 次）
    def t_old(n=20):
        for _ in range(n):
            xb = rt.h2d(np.ascontiguousarray(w_np[ids]))
            tr.head.weight.grad = None
            tr.add_embed_grad(ids, g)
        rt.sync()

    def t_new(n=20):
        for _ in range(n):
            xb, ib = tr.x0_from_ids(ids)
            tr.head.weight.grad = None
            tr.add_embed_grad_device(ib, g_buf)
        rt.sync()

    t_old(3); t_new(3)
    t0 = time.time(); t_old(); dt_old = (time.time() - t0) / 20
    t0 = time.time(); t_new(); dt_new = (time.time() - t0) / 20
    log("旧路径 %.1f ms/步   设备路径 %.1f ms/步   省 %.1f ms（%.1fx）"
        % (dt_old * 1000, dt_new * 1000, (dt_old - dt_new) * 1000, dt_old / max(dt_new, 1e-9)))
    log("结论：%s" % ("PASS（逐位一致）" if (ok_x0 and ok_g) else "FAIL"))
    st.close()
    rt.empty_cache()


if __name__ == "__main__":
    main()
