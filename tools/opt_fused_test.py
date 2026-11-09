# -*- coding: utf-8 -*-
"""A1 验收：融合 fp16-动量 AdamW vs 逐张量路径。

两件事：
  1. **逐位对比**：同样的初始参数/梯度/动量，走 `_step_scalar()` 与 `step()`（融合），
     3 步后比较参数与 m/v 的**位模式**是否完全相等。
  2. **实测计时**：V2.7 28 层，分别用融合与逐张量各跑 3 步，报单步时间。

运行（需独占 GPU）：
    python opt_fused_test.py
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


def bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def parity():
    import tgai_go.train as T
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW

    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    shapes = [(128, 64), (64, 128), (1024,), (7, 3)]
    pa_list, pb_list = [], []
    for shp in shapes:
        w = rng.normal(0, 0.1, shp).astype(np.float32)
        g = rng.normal(0, 0.05, shp).astype(np.float32)
        pa = T.nn.Parameter(w.copy()); pb = T.nn.Parameter(w.copy())
        pa.grad = T.tensor(g.copy()); pb.grad = T.tensor(g.copy())
        pa_list.append(pa); pb_list.append(pb)

    kw = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    oa = Fp16MomAdamW(pa_list, **kw)     # 逐张量
    ob = Fp16MomAdamW(pb_list, **kw)     # 融合
    rt = T._cuda()
    for s in range(3):
        oa.t += 1
        oa._step_scalar()
        ob.step()
    ok = True
    for i, (a, b) in enumerate(zip(pa_list, pb_list)):
        ra = np.asarray(a.to_numpy(), np.float32)
        rb = np.asarray(b.to_numpy(), np.float32)
        same = np.array_equal(bits(ra), bits(rb))
        log("  参数 %d shape=%-12s 位相同=%s  max|Δ|=%.3e"
            % (i, str(ra.shape), same, float(np.abs(ra - rb).max())))
        ok = ok and same
    for i in range(len(pa_list)):
        ma = np.asarray(rt.d2h(oa.m16[i]), np.float32)
        mb = np.asarray(rt.d2h(ob.m16[i]), np.float32)
        va = np.asarray(rt.d2h(oa.v16[i]), np.float32)
        vb = np.asarray(rt.d2h(ob.v16[i]), np.float32)
        same = np.array_equal(bits(ma), bits(mb)) and np.array_equal(bits(va), bits(vb))
        log("  动量 %d 位相同=%s" % (i, same))
        ok = ok and same
    log("逐位对比结论：%s" % ("PASS" if ok else "FAIL"))
    return ok


def timing():
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
    kw = dict(lr=1e-6, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    opts = [Fp16MomAdamW(tr.params_of(i), **kw) for i in range(NL)]
    head = Fp16MomAdamW(tr.head_params(), **kw)
    log("28 层 + 优化器就绪，显存 %d MiB" % RUN.vram())

    seq = 128
    ids = np.random.default_rng(0).integers(0, 30000, seq + 1, dtype=np.int64)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(seq, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32)); sin = T.tensor(np.sin(ang).astype(np.float32))
    x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)

    def full_step(fused):
        xs = tr.forward_all(x0, cos, sin)
        ls, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
        # 逐层反传（梯度）但优化器步骤单独计时
        gs = []
        gg = np.asarray(g, np.float32)
        for i in range(NL - 1, -1, -1):
            x_in = T.tensor(xs[i], requires_grad=True)
            y = tr.fwd(x_in, i, cos, sin)
            (y * T.tensor(gg)).sum().backward()
            if x_in.grad is not None:
                gg = np.asarray(x_in.grad.to_numpy(), np.float32)
            gs.append(i)
        t0 = time.time()
        for o in opts:
            if fused:
                o.step()
            else:
                o.t += 1
                o._step_scalar()
        head.step()
        T._cuda().sync()
        dt = time.time() - t0
        for o in opts:
            o.zero_grad()
        head.zero_grad()
        return dt, float(np.asarray(ls.to_numpy() if hasattr(ls, "to_numpy") else ls))

    for tag, fused in (("逐张量", False), ("融合", True)):
        full_step(fused)
        T._cuda().sync()
        ts = []
        for _ in range(3):
            dt, loss = full_step(fused)
            ts.append(dt)
        log("  %s优化器：单次优化器耗时 中位 %.3f s（loss=%.4f）"
            % (tag, float(np.median(ts)), loss))
    st.close()
    T._cuda().empty_cache()


if __name__ == "__main__":
    ok = parity()
    timing()
    log("A1 结论：%s" % ("PASS" if ok else "FAIL"))
