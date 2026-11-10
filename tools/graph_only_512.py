# -*- coding: utf-8 -*-
"""T=512 只跑图反向的整步（不与 eager 同进程抢显存）。

背景：同进程里先跑一遍 eager 会把池缓存占满，捕获前必须 empty_cache，而之后的重放会撞
rc=700。为避免这个顺序耦合，本脚本只做：前向链 → 捕获 28 张图 → 反复跑图反向整步。
eager 基线用同一 harness、同 seed、同形状单独测过（2038.5 ms，见提交记录）。

运行（需独占 GPU）：python graph_only_512.py
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
SEQ = 512
NL = 28
REP = 3


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
    tr.attach_head(st)

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30000, SEQ + 1, dtype=np.int64)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))
    x0_buf, id_buf = tr.x0_from_ids(ids[:-1])
    x0 = np.asarray(rt.d2h(x0_buf), np.float32)

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

    # 1) 捕获。默认让 LayerGraphBackward 按显存预算自动截断（NG 可显式指定做诊断）。
    NG = int(sys.argv[1]) if len(sys.argv) > 1 else None
    _, xs0 = fwd_chain()
    gb = LayerGraphBackward(tr, cos, sin, seq=SEQ, arena_bytes=480 * 1024 ** 2)
    gb.capture(xs0[:NL]) if NG is None else gb.capture(xs0[:NG])
    log("已捕获 %d 张（T=%d，总层 %d）" % (gb.count, SEQ, NL))

    def step():
        clear()
        ys, xs = fwd_chain()
        loss, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
        gg = np.asarray(g, np.float32)
        for i in range(NL - 1, -1, -1):
            y_t = T.tensor(ys[i], requires_grad=True)
            z = tr.moe_part(y_t, i)
            (z * T.tensor(gg)).sum().backward()
            gw_i = np.asarray(y_t.grad.to_numpy(), np.float32)
            if gb.has(i):
                gg = gb.replay_layer(i, gw_i)
            else:
                xi = T.tensor(xs[i], requires_grad=True)
                ya = tr.fwd_attn(xi, i, cos, sin)
                (ya * T.tensor(gw_i)).sum().backward()
                gg = np.asarray(xi.grad.to_numpy(), np.float32)
        tr.add_embed_grad_device(id_buf, T.tensor(gg)._gpu)
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
    log("T=%d 图反向整步：min %.0f / mean %.0f ms；loss=%.8f"
        % (SEQ, min(ts) * 1000, float(np.mean(ts)) * 1000, loss0))
    gb.close()
    try:
        st.close(); rt.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
