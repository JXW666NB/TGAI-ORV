# -*- coding: utf-8 -*-
"""28 层 attention：逐层捕获图 → 整步对比（eager vs 28 次重放）。

前提：已应用多图补丁（cuda.py 已含 graph_launch_slot / arena-only 分配）。
运行（需独占 GPU）：python graph_28layer.py
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

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"
SEQ = 128
NL = 28
ARENA_PER_GRAPH = 256 * 1024 ** 2      # 每图 256 MiB（先measure再调）


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    D, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = D // NH, float(h["rope_theta"])
    cfg = dict(d_model=D, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    rt = T._cuda()
    if not hasattr(rt, "graph_launch_slot"):
        log("多图补丁未应用"); return 2

    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    log("28 层已装载")

    x_np = np.random.default_rng(0).normal(0, 1, (SEQ, D)).astype(np.float32)
    gw_np = np.random.default_rng(1).normal(0, 1, (SEQ, D)).astype(np.float32)
    x_buf = rt.h2d(x_np)
    gw_buf = rt.h2d(gw_np)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))
    mask = tr._causal_mask(SEQ)
    inv = 1.0 / np.sqrt(DH)

    def block(i):
        L = tr.layers[i]
        x = T.Tensor._from_gpu(x_buf, requires_grad=True)
        hh = L["ln1"](x)
        acc = None
        for t in range(NH):
            q = T._rope_composed(L["q"][t](hh), cos, sin)
            k = T._rope_composed(L["k"][t](hh), cos, sin)
            v = L["v"][t](hh)
            pr = T.functional.softmax((q @ k.T) * inv + mask, axis=-1)
            part = L["o"][t](pr @ v)
            acc = part if acc is None else acc + part
        (acc * T.Tensor._from_gpu(gw_buf)).sum().backward()

    def zero(i):
        for k in ("q", "k", "v", "o"):
            for m in tr.layers[i][k]:
                shp = tuple(int(x) for x in m.weight.shape)
                m.weight.grad = T.tensor(np.zeros(shp, np.float32))

    def sum_grads(i):
        tot = 0.0
        for k in ("q", "k", "v", "o"):
            for m in tr.layers[i][k]:
                tot += float(np.abs(np.asarray(m.weight.grad.to_numpy(), np.float32)).sum())
        return tot

    # eager：28 层全部
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(NL):
        zero(i); block(i)
    torch.cuda.synchronize(); t_eager = time.time() - t0
    eager_sums = [sum_grads(i) for i in range(NL)]
    log("eager 28 层 attention：%.1f ms" % (t_eager * 1000))

    # 逐层捕获（每图独立 arena）
    rt.reserve_capture_arena(ARENA_PER_GRAPH)
    t0 = time.time()
    for i in range(NL):
        zero(i)
        if not (rt.graph_arm() and rt.graph_begin()):
            log("第 %d 层捕获失败" % i); return 2
        block(i)
        if not rt.graph_end():
            log("第 %d 层 graph_end 失败" % i); return 2
    t_cap = time.time() - t0
    off = int(getattr(rt, "_arena_off", 0))
    log("捕获 28 张图耗时 %.1f s；当前 arena 已用 %.1f MiB / %d MiB；总图数=%d"
        % (t_cap, off / 1024 ** 2, ARENA_PER_GRAPH / 1024 ** 2, rt.graph_count()))

    # 重放 28 次
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(NL):
        rt.graph_launch_slot(i)
    torch.cuda.synchronize(); t_graph = time.time() - t0
    replay_sums = [sum_grads(i) for i in range(NL)]

    worst = 0.0
    for i in range(NL):
        worst = max(worst, abs(eager_sums[i] - replay_sums[i]) / max(1e-9, abs(eager_sums[i])))
    log("28 层梯度合计数相对差 最差 %.2e（判据 1e-6）" % worst)
    log("整步（仅 attention 块）：eager %.1f ms → 28 次重放 %.1f ms（%.2fx，省 %.0f ms）"
        % (t_eager * 1000, t_graph * 1000, t_eager / max(t_graph, 1e-9),
           (t_eager - t_graph) * 1000))
    try:
        rt.graph_release_all()
        st.close(); rt.empty_cache()
    except Exception as e:
        log("清理异常（忽略）：%s" % type(e).__name__)
    log("结论：%s" % ("PASS" if worst <= 1e-6 else "FAIL"))
    return 0 if worst <= 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
