# -*- coding: utf-8 -*-
"""整步复测：embedding 路径 numpy vs 设备化，对 28 层单步时间与 loss 的影响。

两条路径除 embedding 外完全一致（同样的 forward_all(_dev)、loss、逐层反传、Fp16MomAdamW）。
判据：loss 相同（相对 ≤1e-6），并给出单步时间。

运行（需独占 GPU）：python full_step_embed.py
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

MODEL_TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
NL = 28
SEQ = 128


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    DM, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = DM // NH, float(h["rope_theta"])
    cfg = dict(d_model=DM, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    tr.attach_head(st)
    emb = st["token_embedding.weight"]
    rt = T._cuda()

    seq = SEQ
    rng = np.random.default_rng(0)
    ids = rng.integers(0, int(h["vocab_size"]) if "vocab_size" in h else 30000, seq + 1,
                       dtype=np.int64)   # seq+1：前向吃 ids[:-1]，targets 用 ids[1:]
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(seq, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32))
    sin = T.tensor(np.sin(ang).astype(np.float32))

    def zero_grads():
        for i in range(NL):
            for p in tr.params_of(i):
                p.grad = None
        for p in tr.head_params():
            p.grad = None

    def run_step(device_embed, x0_np=None, x0_buf=None, id_buf=None):
        zero_grads()
        if device_embed:
            xs = tr.forward_all_dev(x0_buf, cos, sin)
            # 反向循环要用 numpy 形态的 x0（0.5 MB，与 128 MB 的主机传输无关）
            xs[0] = np.asarray(rt.d2h(xs[0]), np.float32)
        else:
            xs = tr.forward_all(x0_np, cos, sin)
        loss, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
        gg = np.asarray(g, np.float32)
        for i in range(NL - 1, -1, -1):
            x_in = T.tensor(xs[i], requires_grad=True)
            y = tr.fwd(x_in, i, cos, sin)
            (y * T.tensor(gg)).sum().backward()
            if x_in.grad is not None:
                gg = np.asarray(x_in.grad.to_numpy(), np.float32)
        if device_embed:
            tr.add_embed_grad_device(id_buf, T.tensor(gg)._gpu)
        else:
            tr.add_embed_grad(ids, gg)
        return float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss))

    # 旧路径
    x0_np = np.ascontiguousarray(np.asarray(emb[ids[:-1]], np.float32))
    l_old = run_step(False, x0_np=x0_np)
    # 新路径
    x0_buf, id_buf = tr.x0_from_ids(ids[:-1])
    l_new = run_step(True, x0_buf=x0_buf, id_buf=id_buf)
    rel = abs(l_old - l_new) / max(1e-9, abs(l_old))
    log("loss: numpy 路径 %.8f   设备路径 %.8f   相对差 %.2e" % (l_old, l_new, rel))

    def timeit(device_embed, rep=3):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(rep):
            if device_embed:
                run_step(True, x0_buf=x0_buf, id_buf=id_buf)
            else:
                run_step(False, x0_np=x0_np)
        torch.cuda.synchronize()
        return (time.time() - t0) / rep

    t_old = timeit(False)
    t_new = timeit(True)
    log("单步（前反向+embedding，不含优化器）: numpy %.3f s   设备 %.3f s   省 %.0f ms"
        % (t_old, t_new, (t_old - t_new) * 1000))
    log("结论：%s" % ("loss 一致（≤1e-6）" if rel <= 1e-6 else "loss 不一致，需排查"))
    st.close()
    rt.empty_cache()


if __name__ == "__main__":
    main()
