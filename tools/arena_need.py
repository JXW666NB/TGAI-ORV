# -*- coding: utf-8 -*-
"""量 T=512 下"每张图需要多少捕获 arena"，并测出 32 GB 卡上最多能放多少张。

运行（需独占 GPU）：python arena_need.py
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
    rt = T._cuda()
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    free0 = rt.free_device_bytes() / (1 << 30)
    log("装载 28 层后设备空闲 %.2f GiB" % free0)

    rng = np.random.default_rng(0)
    x_np = rng.normal(0, 1, (SEQ, D)).astype(np.float32)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))

    x_buf = rt.h2d(x_np)
    gw_buf = rt.h2d(np.zeros((SEQ, D), np.float32))
    rt.reserve_capture_arena(2 * 1024 ** 3)          # 先给 2 GiB，看一层用多少
    for k in ("q", "k", "v", "o"):
        for m in tr.layers[0][k]:
            m.weight.grad = T.tensor(np.zeros(tuple(int(x) for x in m.weight.shape), np.float32))
    ok = rt.graph_arm() and rt.graph_begin()
    log("graph_begin=%s" % ok)
    xi = T.Tensor._from_gpu(x_buf, requires_grad=True)
    ya = tr.fwd_attn(xi, 0, cos, sin)
    (ya * T.Tensor._from_gpu(gw_buf)).sum().backward()
    log("graph_end=%s" % rt.graph_end())
    used = int(getattr(rt, "_arena_off", 0))
    log("T=%d 单层图 arena 用量：%.1f MiB" % (SEQ, used / 1024 ** 2))
    free1 = rt.free_device_bytes() / (1 << 30)
    log("一层捕获后设备空闲 %.2f GiB ⇒ 按此推算最多可放约 %d 张（每张 %.0f MiB）"
        % (free1, int((free1 - 0.5) * 1024 ** 3 / max(used, 1)), used / 1024 ** 2))
    try:
        rt.graph_release_all(); st.close(); rt.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
