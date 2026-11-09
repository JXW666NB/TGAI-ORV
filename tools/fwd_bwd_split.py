# -*- coding: utf-8 -*-
"""拆分测量：28 层 attention 的**前向**与**前向+反向**各占多少。

用途：如果只把反向做成图（前向保持 eager，因为 MoE 交错），能省的是反向那部分。
运行（需独占 GPU）：python fwd_bwd_split.py
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
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)

    x_np = np.random.default_rng(0).normal(0, 1, (SEQ, D)).astype(np.float32)
    gw_np = np.random.default_rng(1).normal(0, 1, (SEQ, D)).astype(np.float32)
    x_buf = T._cuda().h2d(x_np)
    gw_buf = T._cuda().h2d(gw_np)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))
    mask = tr._causal_mask(SEQ)
    inv = 1.0 / np.sqrt(DH)

    def attn(i, back):
        L = tr.layers[i]
        x = T.Tensor._from_gpu(x_buf, requires_grad=back)
        hh = L["ln1"](x)
        acc = None
        for t in range(NH):
            q = T._rope_composed(L["q"][t](hh), cos, sin)
            k = T._rope_composed(L["k"][t](hh), cos, sin)
            v = L["v"][t](hh)
            pr = T.functional.softmax((q @ k.T) * inv + mask, axis=-1)
            part = L["o"][t](pr @ v)
            acc = part if acc is None else acc + part
        if back:
            (acc * T.Tensor._from_gpu(gw_buf)).sum().backward()
        return acc

    def zero(i):
        for k in ("q", "k", "v", "o"):
            for m in tr.layers[i][k]:
                shp = tuple(int(x) for x in m.weight.shape)
                m.weight.grad = T.tensor(np.zeros(shp, np.float32))

    # 预热
    for i in range(2):
        zero(i); attn(i, True)
    torch.cuda.synchronize()

    torch.cuda.synchronize(); t0 = time.time()
    for i in range(NL):
        attn(i, False)
    torch.cuda.synchronize(); t_fwd = time.time() - t0

    torch.cuda.synchronize(); t0 = time.time()
    for i in range(NL):
        zero(i); attn(i, True)
    torch.cuda.synchronize(); t_both = time.time() - t0

    log("28 层 attention 仅前向：%.1f ms" % (t_fwd * 1000))
    log("28 层 attention 前向+反向：%.1f ms" % (t_both * 1000))
    log("⇒ 反向部分约 %.1f ms（占 %.0f%%）" % ((t_both - t_fwd) * 1000,
                                             100.0 * (t_both - t_fwd) / max(t_both, 1e-9)))
    try:
        st.close(); T._cuda().empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
