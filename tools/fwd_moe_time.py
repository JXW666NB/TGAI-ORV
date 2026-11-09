# -*- coding: utf-8 -*-
"""量出 28 层**前向（含 MoE）**耗时，用于把整步拆成三段实测：前向 / 反向 / 优化器。

运行（需独占 GPU）：python fwd_moe_time.py
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
    tr.attach_head(st)

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30000, SEQ + 1, dtype=np.int64)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))
    x0 = np.ascontiguousarray(np.asarray(st["token_embedding.weight"][ids[:-1]], np.float32))

    # 预热
    tr.forward_all(x0, cos, sin)
    torch.cuda.synchronize()

    torch.cuda.synchronize(); t0 = time.time()
    xs = tr.forward_all(x0, cos, sin)
    torch.cuda.synchronize(); t_fwd = time.time() - t0
    log("28 层前向（逐层 fwd，含 MoE）：%.1f ms；得到 %d 个层输入" % (t_fwd * 1000, len(xs)))

    # loss 头（对照用）
    torch.cuda.synchronize(); t0 = time.time()
    loss, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
    torch.cuda.synchronize(); t_h = time.time() - t0
    lv = float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss))
    log("head+loss 反向：%.1f ms（loss=%.6f）" % (t_h * 1000, lv))
    try:
        st.close(); T._cuda().empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
