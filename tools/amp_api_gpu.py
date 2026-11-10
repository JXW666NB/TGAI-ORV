# -*- coding: utf-8 -*-
"""GPU 上验证 T.amp()：fp16 GEMM 生效（速度）+ 真实训练步数值有限（含梯度检查）。

运行（需独占 GPU）：python amp_api_gpu.py
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
D, NH, DF, NE, TOPK, THETA = 1024, 16, 4096, 8, 2, 1e4
DH = D // NH


def _np(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    T.set_default_device("cuda")
    rt = T._cuda()
    ok = True

    # 1) 纯 GEMM：amp 开/关对比
    rng = np.random.default_rng(0)
    a_np = rng.normal(0, 0.1, (2048, 2048)).astype(np.float32)
    b_np = rng.normal(0, 0.1, (2048, 6144)).astype(np.float32)

    def bench():
        a = T.tensor(a_np)
        b = T.tensor(b_np)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(10):
            a @ b
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / 10.0

    log("amp 关: %s" % (T.amp_state(),))
    t_off = bench()
    with T.amp():
        log("amp 开: %s" % (T.amp_state(),))
        t_on = bench()
    log("GEMM 2048x2048x6144: fp32 %.2f ms → fp16 %.2f ms（%.2fx）" % (t_off, t_on, t_off / max(t_on, 1e-9)))
    ok = ok and t_on < t_off * 0.5
    log("AMP 关闭后状态: %s" % (T.amp_state(),))

    # 2) 真实训练步（8 层）：amp 下 loss 与梯度必须有限
    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    cfg = dict(d_model=D, n_heads=NH, d_ff=DF, n_experts=NE, n_activated=TOPK,
               n_layers=int(h["n_layers"]), rope_theta=THETA)
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=8)
    tr.load_all(st)
    seq = 128
    x = rng.normal(0, 1, (seq, D)).astype(np.float32)
    gw = rng.normal(0, 1, (seq, D)).astype(np.float32)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(seq, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))

    def step():
        for i in range(8):
            for p in tr.params_of(i):
                p.grad = None
        cur = T.tensor(x, requires_grad=True)
        for i in range(8):
            cur = tr.fwd(cur, i, cos, sin)
        loss = (cur * T.tensor(gw)).sum()
        loss.backward()
        gsum = 0.0
        for i in range(8):
            for p in tr.params_of(i):
                if p.grad is not None:
                    gsum += float(np.abs(_np(p.grad)).sum())
        return float(_np(loss)), gsum

    l32, g32 = step()
    with T.amp():
        l16, g16 = step()
    log("真实步（8 层）：fp32 loss %.4f / 梯度合计 %.4g；fp16 loss %.4f / 梯度合计 %.4g"
        % (l32, g32, l16, g16))
    finite = np.isfinite([l16, g16]).all()
    log("fp16 步数值有限：%s；loss 相对差 %.2e" % (finite, abs(l16 - l32) / max(1e-9, abs(l32))))
    ok = ok and bool(finite) and abs(l16 - l32) / max(1e-9, abs(l32)) < 1e-2

    try:
        st.close(); rt.empty_cache()
    except Exception:
        pass
    log("总结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
