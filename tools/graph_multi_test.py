# -*- coding: utf-8 -*-
"""验证多图支持：捕获两张图（两层各一张），分别重放，各自与 eager 逐位比较。

前提：已应用 python tools/apply_multigraph.py
运行（需独占 GPU）：python graph_multi_test.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"
SEQ = 128
NL = 2


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
        log("多图支持未应用（没有 graph_launch_slot）")
        return 2

    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)

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

    def grads(i):
        L = tr.layers[i]
        out = {}
        for key, axis in (("q", 0), ("k", 0), ("v", 0), ("o", 1)):
            gs = [np.asarray(m.weight.grad.to_numpy(), np.float32) for m in L[key]]
            out[key] = np.concatenate(gs, axis=axis) if len(gs) > 1 else gs[0]
        return out

    # 每层的 eager 结果
    eager = {}
    for i in range(NL):
        zero(i); block(i); rt.sync()
        eager[i] = grads(i)

    # 逐层捕获（每层一张图）
    rt.reserve_capture_arena(4 * 1024 ** 3)
    for i in range(NL):
        zero(i)
        if not (rt.graph_arm() and rt.graph_begin()):
            log("第 %d 层 graph_begin 失败" % i); return 2
        block(i)
        if not rt.graph_end():
            log("第 %d 层 graph_end 失败" % i); return 2
        log("第 %d 层捕获完成，累计图数=%d" % (i, rt.graph_count()))

    log("总图数: %d（期望 %d）" % (rt.graph_count(), NL))
    ok_all = True
    for i in range(NL):
        # 注意：**不能在重放前 zero(i)** —— 那会替换 .grad 对象，图仍写捕获期的旧缓冲，
        # 读到的就成了新零张量（这就是之前 1.44/1.71 的来源）。
        if not rt.graph_launch_slot(i):
            log("第 %d 层重放失败" % i); ok_all = False; continue
        rt.sync()
        g = grads(i)
        worst = 0.0
        for key in ("q", "k", "v", "o"):
            worst = max(worst, float(np.abs(g[key] - eager[i][key]).max() /
                                     max(1e-9, float(np.abs(eager[i][key]).max()))))
        log("第 %d 层：图重放 vs eager 最差 %.2e（判据 1e-6）" % (i, worst))
        ok_all = ok_all and worst <= 1e-6

    try:
        st.close(); rt.empty_cache()
    except Exception as e:
        log("清理异常（忽略）：%s" % type(e).__name__)
    log("结论：%s" % ("PASS（多图逐位一致）" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
