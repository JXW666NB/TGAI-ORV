# -*- coding: utf-8 -*-
"""二分诊断：T=512 下只捕获 2 层图，验证重放与读梯度是否正常。

如果 2 层 OK、28 层 rc=700 ⇒ 问题在规模/显存，而非 T=512 的图机制本身。
运行（需独占 GPU）：python graph_2layer_512.py
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
SEQ = 512
NL = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def main():
    import torch
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
    rt = T._cuda()
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)

    rng = np.random.default_rng(0)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))
    xs = [rng.normal(0, 1, (SEQ, D)).astype(np.float32) for _ in range(NL)]
    gws = [rng.normal(0, 1, (SEQ, D)).astype(np.float32) for _ in range(NL)]

    def zero(i):
        for k in ("q", "k", "v", "o"):
            for m in tr.layers[i][k]:
                m.weight.grad = T.tensor(np.zeros(tuple(int(x) for x in m.weight.shape), np.float32))

    # eager 参照
    e_g, e_dx = [], []
    for i in range(NL):
        zero(i)
        xi = T.tensor(xs[i], requires_grad=True)
        ya = tr.fwd_attn(xi, i, cos, sin)
        (ya * T.tensor(gws[i])).sum().backward()
        e_dx.append(np.asarray(xi.grad.to_numpy(), np.float32))
        e_g.append([np.asarray(m.weight.grad.to_numpy(), np.float32)
                    for k in ("q", "k", "v", "o") for m in tr.layers[i][k]])
    rt.sync()

    gb = LayerGraphBackward(tr, cos, sin, seq=SEQ, arena_bytes=512 * 1024 ** 2)
    n = gb.capture(xs)
    log("捕获 %d 张（T=%d）" % (n, SEQ))
    worst = 0.0
    for i in range(NL):
        dx = gb.replay_layer(i, gws[i])          # 逐层重放（与训练循环一致）
        rg = [[np.asarray(t.to_numpy(), np.float32) for t in gb._grad_refs[i]]][0]
        for a, b in zip(e_g[i], rg):
            worst = max(worst, float(np.abs(a - b).max() / max(1e-9, float(np.abs(a).max()))))
        a, b = e_dx[i], dx
        worst = max(worst, float(np.abs(a - b).max() / max(1e-9, float(np.abs(a).max()))))
        log("  第 %d 层重放+读梯度 OK" % i)
    log("T=%d 2 层：参数梯度+dL/dx 最大相对差 %.2e ⇒ %s"
        % (SEQ, worst, "PASS" if worst <= 1e-6 else "FAIL"))
    gb.close()
    try:
        st.close(); rt.empty_cache()
    except Exception:
        pass
    return 0 if worst <= 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
