# -*- coding: utf-8 -*-
"""A2 判定实验：把**一层**的前向+反向捕获成 CUDA Graph，重放并与逐算子路径对比。

要回答三件事：
  1. 我们的层（含 MoE/attention）能否被捕获（图里不能有 D2H/同步）；
  2. 重放结果与逐算子路径是否**逐位一致**；
  3. 每层耗时能降多少（当前实测 25.5 ms/层）。

运行（需独占 GPU）：python graph_layer_test.py
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
    DM, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = DM // NH, float(h["rope_theta"])
    cfg = dict(d_model=DM, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    emb = st["token_embedding.weight"]
    rt = T._cuda()
    log("graph 可用: %s" % rt.graph_available())

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30000, SEQ, dtype=np.int64)
    x0 = np.ascontiguousarray(emb[ids], np.float32)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32))
    sin = T.tensor(np.sin(ang).astype(np.float32))
    in_buf = rt.h2d(x0)                      # 固定输入缓冲
    g_buf = rt.h2d(rng.normal(0, 1, (SEQ, DM)).astype(np.float32))
    params = tr.params_of(0)

    def zero_grads():
        for p in params:
            p.grad = None

    def step():
        zero_grads()
        x = T.Tensor._from_gpu(in_buf, requires_grad=True)
        y = tr.fwd(x, 0, cos, sin)
        (y * T.Tensor._from_gpu(g_buf)).sum().backward()

    def grads_np():
        out = []
        for p in params:
            out.append(np.asarray(p.grad.to_numpy() if hasattr(p.grad, "to_numpy") else p.grad,
                                  np.float32).copy())
        return out

    step()                                   # 预热（NVRTC/cuBLAS）
    rt.sync()
    base = grads_np()
    t0 = time.time()
    for _ in range(3):
        step()
    rt.sync()
    t_plain = (time.time() - t0) / 3.0
    base2 = grads_np()

    ok_arm = rt.graph_arm()
    log("graph_arm: %s" % ok_arm)
    ok_begin = rt.graph_begin()
    log("graph_begin: %s" % ok_begin)
    if not ok_begin:
        log("捕获不可用，A2 需要改用 graph_train.GraphStep 的宿主缓冲方案")
        st.close()
        return
    step()                                  # 捕获期间会真实执行一次
    ok_end = rt.graph_end()
    log("graph_end: %s" % ok_end)
    if not ok_end:
        st.close()
        return

    rt.graph_launch()                        # 第一次重放
    rt.sync()
    rep = grads_np()
    t0 = time.time()
    for _ in range(3):
        rt.graph_launch()
    rt.sync()
    t_graph = (time.time() - t0) / 3.0

    worst = 0.0
    for a, b, c in zip(base, rep, base2):
        d = max(float(np.abs(a - b).max()) / max(1e-9, float(np.abs(a).max())),
                float(np.abs(b - c).max()) / max(1e-9, float(np.abs(c).max())))
        worst = max(worst, d)
    log("逐算子 %.1f ms   图重放 %.1f ms   加速 %.2fx   梯度最大相对差 %.2e"
        % (t_plain * 1000, t_graph * 1000, t_plain / max(t_graph, 1e-9), worst))
    log("A2 判定：%s" % ("PASS（可捕获且一致）" if worst <= 1e-6 else "FAIL（不一致）"))
    st.close()
    rt.empty_cache()


if __name__ == "__main__":
    main()
