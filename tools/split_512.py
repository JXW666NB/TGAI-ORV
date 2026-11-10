# -*- coding: utf-8 -*-
"""T=512、28 层的整步分解：前向链 / MoE 反向 / 注意力反向 各占多少。

为什么要拆：目标是"T=512 整步 ≤ 0.911 s"，当前图反向方案 1165 ms（16/28 层有图）。
不知道 1165 ms 花在哪就只是在猜。这个脚本把四段分别计时（CUDA event，各段单独跑）。

运行（需独占 GPU）：python split_512.py
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
SEQ, NL = 512, 28
REP = 3


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

    def ev(fn, rep=REP):
        fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(rep):
            fn()
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / rep

    # 1) 前向链（attn + MoE，28 层）
    def fwd():
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

    t_fwd = ev(fwd)
    ys, xs = fwd()
    log("前向链（28 层 attn+MoE）：%.1f ms" % t_fwd)

    # 2) 只有 MoE 反向
    gw = [rng.normal(0, 1, (SEQ, D)).astype(np.float32) for _ in range(NL)]

    def clear():
        for i in range(NL):
            for k in ("q", "k", "v", "o"):
                for m in tr.layers[i][k]:
                    m.weight.grad = None
        for p in tr.head_params():
            p.grad = None

    def moe_bwd():
        clear()
        for i in range(NL):
            y_t = T.tensor(ys[i], requires_grad=True)
            z = tr.moe_part(y_t, i)
            (z * T.tensor(gw[i])).sum().backward()

    t_moe = ev(moe_bwd)
    log("MoE 反向（28 层）：%.1f ms" % t_moe)

    # 3) 只有注意力反向（逐算子）
    def attn_bwd():
        clear()
        for i in range(NL):
            xi = T.tensor(xs[i], requires_grad=True)
            ya = tr.fwd_attn(xi, i, cos, sin)
            (ya * T.tensor(gw[i])).sum().backward()

    t_attn = ev(attn_bwd)
    log("注意力反向（28 层，逐算子）：%.1f ms" % t_attn)

    # 4) 词表头（loss + 反向）
    def head():
        clear()
        tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)

    t_head = ev(head)
    log("词表头 loss+反向：%.1f ms" % t_head)

    log("合计（前向 %.0f + MoE反 %.0f + 注意反 %.0f + 头 %.0f）= %.0f ms"
        % (t_fwd, t_moe, t_attn, t_head, t_fwd + t_moe + t_attn + t_head))
    log("其中注意力反向若全 28 层走图（按 7.24× 估算）：%.0f ms ⇒ 整步约 %.0f ms"
        % (t_attn / 7.24, t_fwd + t_moe + t_attn / 7.24 + t_head))
    try:
        st.close(); rt.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
