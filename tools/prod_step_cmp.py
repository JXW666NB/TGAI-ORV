# -*- coding: utf-8 -*-
"""产线口径整步：embedding(设备) → forward_all → loss → 反向(eager 或 图) → 融合优化器。

只比时间（权重会随 opt.step 漂移，loss 对照已在 graph_whole_step.py 做过 0.00e+00）。
运行（需独占 GPU）：python prod_step_cmp.py
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
REP = 3


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.layer_graph import LayerGraphBackward
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW
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
    tr.attach_head(st)
    params = []
    for i in range(NL):
        params += tr.params_of(i)
    params += list(tr.head_params())
    opt = Fp16MomAdamW(params, lr=1e-5)

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30000, SEQ + 1, dtype=np.int64)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))

    x0_buf, id_buf = tr.x0_from_ids(ids[:-1])
    x0 = np.asarray(T._cuda().d2h(x0_buf), np.float32)

    def zero():
        for i in range(NL):
            for k in ("q", "k", "v", "o"):
                for m in tr.layers[i][k]:
                    m.weight.grad = None
        for p in tr.head_params():
            p.grad = None

    def fwd_chain():
        ys, xs = [], [x0]
        x = T.tensor(x0)
        for i in range(NL):
            ya = tr.fwd_attn(x, i, cos, sin)
            ys.append(np.asarray(ya.to_numpy(), np.float32))
            y = tr.moe_part(ya, i)
            xn = np.asarray(y.to_numpy(), np.float32)
            xs.append(xn); x = T.tensor(xn)
        return ys, xs

    def step_eager():
        zero()
        ys, xs = fwd_chain()
        loss, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
        gg = np.asarray(g, np.float32)
        for i in range(NL - 1, -1, -1):
            xi = T.tensor(xs[i], requires_grad=True)
            ya = tr.fwd_attn(xi, i, cos, sin)
            z = tr.moe_part(ya, i)
            (z * T.tensor(gg)).sum().backward()
            gg = np.asarray(xi.grad.to_numpy(), np.float32)
        tr.add_embed_grad_device(id_buf, T.tensor(gg)._gpu)
        opt.step()
        return float(np.asarray(loss.to_numpy()))

    # 捕获图（需要一次前向来拿每层输入）
    ys0, xs0 = fwd_chain()
    gb = LayerGraphBackward(tr, cos, sin, seq=SEQ, arena_bytes=128 * 1024 ** 2)
    gb.capture(xs0[:NL])
    log("已捕获 %d 张图" % gb.count() if hasattr(gb, "count") else "已捕获图")

    def step_graph():
        zero()
        ys, xs = fwd_chain()
        loss, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
        gg = np.asarray(g, np.float32)
        gws = [None] * NL
        for i in range(NL - 1, -1, -1):
            y_t = T.tensor(ys[i], requires_grad=True)
            z = tr.moe_part(y_t, i)
            (z * T.tensor(gg)).sum().backward()
            gg = gb.replay_layer(i, np.asarray(y_t.grad.to_numpy(), np.float32))
        dx0 = gg
        tr.add_embed_grad_device(id_buf, T.tensor(dx0)._gpu)
        opt.step()
        return float(np.asarray(loss.to_numpy()))

    def timeit(fn):
        fn(); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(REP):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / REP

    t_e = timeit(step_eager)
    t_g = timeit(step_graph)
    log("产线口径整步（含 embedding 设备化 + 融合优化器）：eager %.1f ms → graph %.1f ms（%.2fx，省 %.0f ms）"
        % (t_e * 1000, t_g * 1000, t_e / max(t_g, 1e-9), (t_e - t_g) * 1000))
    log("目标 ≤1.0 s：%s" % ("达成" if t_g <= 1.0 else "未达成（%.2f s）" % t_g))
    gb.close()
    try:
        st.close(); T._cuda().empty_cache()
    except Exception:
        pass
    return 0 if t_g <= 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
