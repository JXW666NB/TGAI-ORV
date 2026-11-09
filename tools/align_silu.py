# -*- coding: utf-8 -*-
"""定点核对：scale=8（误差 4.6e-2 那个点）下，把专家拆成 4 段逐一比。

    gate(x) → up(x) → silu(gate)*up → @down.T

哪一段开始爆，就是根因所在（预期是 silu 的大幅值分支）。

运行：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_silu.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
MODEL_PT = r"D:\TGAI\models\V2.7\tgai_v27_v5_merged.pt"
SEQ, D = 64, 1024


def relerr(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def silu_np(x):
    return x / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    cfg = dict(d_model=int(h["d_model"]), n_heads=int(h["n_heads"]), d_ff=int(h["d_ff"]),
               n_experts=int(h["n_experts"]), n_activated=int(h["n_activated"]),
               n_layers=int(h["n_layers"]), rope_theta=float(h["rope_theta"]))
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    sd = torch.load(MODEL_PT, map_location="cpu", mmap=True, weights_only=False)["model_state_dict"]
    p0 = "blocks.0.moe.experts.0."
    Wg = np.asarray(sd[p0 + "w1.weight"].float().numpy(), np.float32)
    Wu = np.asarray(sd[p0 + "w2.weight"].float().numpy(), np.float32)
    Wd = np.asarray(sd[p0 + "w3.weight"].float().numpy(), np.float32)
    expert = tr.layers[0]["moe"].experts[0]

    base = np.random.default_rng(0).normal(0, 1, (SEQ, D)).astype(np.float32)
    for scale in (1.0, 4.0, 8.0):
        Y = (base * scale).astype(np.float32)
        g_np = Y @ Wg.T
        u_np = Y @ Wu.T
        s_np = silu_np(g_np)
        h_np = s_np * u_np
        o_np = h_np @ Wd.T
        g_o = np.asarray(expert.gate(T.tensor(Y)).to_numpy(), np.float32)
        u_o = np.asarray(expert.up(T.tensor(Y)).to_numpy(), np.float32)
        # silu 单独比：把 ORV 的 gate 输出喂给 ORV 的 silu，再与 numpy silu(g_np) 比
        s_o = np.asarray(T.functional.silu(T.tensor(g_np)).to_numpy(), np.float32)
        o_o = np.asarray(expert.down(T.tensor(h_np)).to_numpy(), np.float32)
        log("scale=%-4g |Y|max=%8.2f |g|max=%9.2f" % (scale, float(np.abs(Y).max()), float(np.abs(g_np).max())))
        log("   gate(x)      %s  %.2e" % ("一致" if relerr(g_o, g_np) < 1e-5 else "★分叉", relerr(g_o, g_np)))
        log("   up(x)        %s  %.2e" % ("一致" if relerr(u_o, u_np) < 1e-5 else "★分叉", relerr(u_o, u_np)))
        log("   silu(g)      %s  %.2e" % ("一致" if relerr(s_o, s_np) < 1e-5 else "★分叉", relerr(s_o, s_np)))
        log("   h=down(...)  %s  %.2e" % ("一致" if relerr(o_o, o_np) < 1e-5 else "★分叉", relerr(o_o, o_np)))
    st.close()


if __name__ == "__main__":
    main()
