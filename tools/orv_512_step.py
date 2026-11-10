# -*- coding: utf-8 -*-
"""T=512 的 ORV 整步：融合优化器 + （可选）逐层图反向，CUDA event 计时。

目的：回答"加速做没做"——与本会话之前那张表（T=512，ORV 1.49-1.87 s）同条件复测。
显存账：3B fp32 权重 ~12 GiB（.tg bf16 载入后按 fp32 计算）+ fp16 AdamW m/v ~11 GiB。
28 张图各自缓存 T=512 的中间量，与优化器同时开会超 32 GB，因此：
  * eager 步：含优化器（真实训练形态）
  * graph 步：不含优化器（纯前反向），并单独给出"优化器已实测 47 ms"
运行（需独占 GPU）：python orv_512_step.py
"""
import ctypes
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
SEQ = 512
NL = 28
REP = 3


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.layer_graph import LayerGraphBackward
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    with_grad = "--with-opt" in sys.argv
    with_graph = "--with-graph" in sys.argv

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    D, NHh = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = D // NHh, float(h["rope_theta"])
    cfg = dict(d_model=D, n_heads=NHh, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    rt = T._cuda()
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    tr.attach_head(st)

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30000, SEQ + 1, dtype=np.int64)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))
    x0_buf, id_buf = tr.x0_from_ids(ids[:-1])
    x0 = np.asarray(rt.d2h(x0_buf), np.float32)

    opt = None
    if with_grad:
        params = []
        for i in range(NL):
            params += tr.params_of(i)
        params += list(tr.head_params())
        opt = Fp16MomAdamW(params, lr=1e-5)

    def clear():
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
            xs.append(xn)
            x = T.tensor(xn)
        return ys, xs

    gb = None
    if with_graph:
        ys0, xs0 = fwd_chain()
        gb = LayerGraphBackward(tr, cos, sin, seq=SEQ, arena_bytes=384 * 1024 ** 2)
        gb.capture(xs0[:NL])
        log("已捕获 %d 张图" % NL)

    def step():
        clear()
        ys, xs = fwd_chain()
        loss, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
        gg = np.asarray(g, np.float32)
        if gb is None:
            for i in range(NL - 1, -1, -1):
                xi = T.tensor(xs[i], requires_grad=True)
                ya = tr.fwd_attn(xi, i, cos, sin)
                z = tr.moe_part(ya, i)
                (z * T.tensor(gg)).sum().backward()
                gg = np.asarray(xi.grad.to_numpy(), np.float32)
        else:
            for i in range(NL - 1, -1, -1):
                y_t = T.tensor(ys[i], requires_grad=True)
                z = tr.moe_part(y_t, i)
                (z * T.tensor(gg)).sum().backward()
                gg = gb.replay_layer(i, np.asarray(y_t.grad.to_numpy(), np.float32))
        tr.add_embed_grad_device(id_buf, T.tensor(gg)._gpu)
        if opt is not None:
            opt.step()
        return float(np.asarray(loss.to_numpy()))

    loss0 = step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REP):
        torch.cuda.synchronize()
        t0 = time.time()
        step()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    tag = "eager" + ("+opt" if opt is not None else "") + ("+graph" if gb else "")
    log("T=%d 28 层整步（%s）：min %.0f / mean %.0f ms；loss=%.6f"
        % (SEQ, tag, min(ts) * 1000, float(np.mean(ts)) * 1000, loss0))
    if gb is not None:
        gb.close()
    try:
        st.close(); rt.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
