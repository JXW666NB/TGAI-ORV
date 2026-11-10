# -*- coding: utf-8 -*-
"""验证 layer_graph.LayerGraphBackward 类：捕获 2 层、重放、与 eager 逐位比较。

运行（需独占 GPU）：python layer_graph_api_test.py
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
SEQ = 64
NL = 2


def main():
    import tgai_go.train as T
    from tgai_go.train.layer_graph import LayerGraphBackward
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
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)

    rng = np.random.default_rng(0)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))
    xs = [rng.normal(0, 1, (SEQ, D)).astype(np.float32) for _ in range(NL)]
    gws = [rng.normal(0, 1, (SEQ, D)).astype(np.float32) for _ in range(NL)]

    # eager 结果
    e_grads, e_dx = [], []
    for i in range(NL):
        for k in ("q", "k", "v", "o"):
            for m in tr.layers[i][k]:
                m.weight.grad = T.tensor(np.zeros(tuple(int(x) for x in m.weight.shape), np.float32))
        xi = T.tensor(xs[i], requires_grad=True)
        ya = tr.fwd_attn(xi, i, cos, sin)
        (ya * T.tensor(gws[i])).sum().backward()
        e_dx.append(np.asarray(xi.grad.to_numpy(), np.float32))
        e_grads.append([np.asarray(m.weight.grad.to_numpy(), np.float32)
                        for k in ("q", "k", "v", "o") for m in tr.layers[i][k]])
    T._cuda().sync()

    gb = LayerGraphBackward(tr, cos, sin, seq=SEQ)
    n = gb.capture(xs)
    log("捕获 %d 张图" % n)
    dx = gb.replay(gws)
    g_graph = gb.grads_numpy()

    worst = 0.0
    for i in range(NL):
        for a, b in zip(e_grads[i], g_graph[i]):
            worst = max(worst, float(np.abs(a - b).max() / max(1e-9, float(np.abs(a).max()))))
        a, b = e_dx[i], dx[i]
        worst = max(worst, float(np.abs(a - b).max() / max(1e-9, float(np.abs(a).max()))))
    log("类 API：参数梯度 + dL/dx 最大相对差 %.2e（判据 1e-6）" % worst)
    gb.close()
    try:
        st.close(); T._cuda().empty_cache()
    except Exception:
        pass
    ok = worst <= 1e-6
    log("结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
