# -*- coding: utf-8 -*-
"""28 层「反向阶段」集成验证：eager 重算+反传  vs  逐层图重放。

真实数据构造：先 eager 前向跑完整 28 层（fwd_attn + moe_part）拿到每层输入 xs[i]；
每层梯度用固定随机 gw_i（两次对照用同一份，公平）；参数梯度逐位比较。
运行（需独占 GPU）：python graph_28_backward.py
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
ARENA = 256 * 1024 ** 2


def main():
    import ctypes
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

    rng = np.random.default_rng(0)
    x_np = rng.normal(0, 1, (SEQ, D)).astype(np.float32)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))

    # 1) eager 前向链，拿每层输入
    xs = [x_np]
    x = T.tensor(x_np)
    for i in range(NL):
        y = tr.fwd(x, i, cos, sin)
        xn = np.asarray(y.to_numpy(), np.float32)
        xs.append(xn)
        x = T.tensor(xn)
    rt.sync()
    log("前向链完成，得到 %d 个层输入" % (len(xs) - 1))

    gws = [rng.normal(0, 1, (SEQ, D)).astype(np.float32) for _ in range(NL)]

    def zero(i):
        for k in ("q", "k", "v", "o"):
            for m in tr.layers[i][k]:
                shp = tuple(int(x) for x in m.weight.shape)
                m.weight.grad = T.tensor(np.zeros(shp, np.float32))

    def gsum(i):
        return sum(float(np.abs(np.asarray(m.weight.grad.to_numpy(), np.float32)).sum())
                   for k in ("q", "k", "v", "o") for m in tr.layers[i][k])

    # 2) eager 反向（重算 + 反传）
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(NL - 1, -1, -1):
        zero(i)
        xi = T.tensor(xs[i], requires_grad=True)
        y = tr.fwd_attn(xi, i, cos, sin)
        (y * T.tensor(gws[i])).sum().backward()
    torch.cuda.synchronize(); t_eager = time.time() - t0
    e_sums = [gsum(i) for i in range(NL)]
    log("eager 反向：%.1f ms" % (t_eager * 1000))

    # 3) 逐层捕获（输入/梯度缓冲固定）
    x_bufs = [rt.h2d(xs[i]) for i in range(NL)]
    gw_bufs = [rt.h2d(gws[i]) for i in range(NL)]
    rt.reserve_capture_arena(ARENA)
    t0 = time.time()
    for i in range(NL):
        zero(i)
        if not (rt.graph_arm() and rt.graph_begin()):
            log("第 %d 层 graph_begin 失败" % i); return 2
        xi = T.Tensor._from_gpu(x_bufs[i], requires_grad=True)
        y = tr.fwd_attn(xi, i, cos, sin)
        (y * T.Tensor._from_gpu(gw_bufs[i])).sum().backward()
        if not rt.graph_end():
            log("第 %d 层 graph_end 失败" % i); return 2
    t_cap = time.time() - t0
    log("捕获 28 张图：%.1f s；图数=%d" % (t_cap, rt.graph_count()))

    def h2d_into(buf, arr):
        a = np.ascontiguousarray(arr, np.float32)
        rt._activate()
        rt._chk(rt.cu.cuMemcpyHtoD(ctypes.c_uint64(int(buf.ptr)),
                                   a.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_size_t(a.nbytes)), "cuMemcpyHtoD")

    # 4) 图反向（每层写梯度缓冲后重放）
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(NL - 1, -1, -1):
        h2d_into(gw_bufs[i], gws[i])
        rt.graph_launch_slot(i)
    torch.cuda.synchronize(); t_graph = time.time() - t0
    r_sums = [gsum(i) for i in range(NL)]

    worst = max(abs(e_sums[i] - r_sums[i]) / max(1e-9, abs(e_sums[i])) for i in range(NL))
    log("28 层参数梯度合计 最大相对差 %.2e（判据 1e-6）" % worst)
    log("反向阶段：eager %.1f ms → 图重放 %.1f ms（%.2fx，省 %.0f ms）"
        % (t_eager * 1000, t_graph * 1000, t_eager / max(t_graph, 1e-9),
           (t_eager - t_graph) * 1000))
    try:
        rt.graph_release_all(); st.close(); rt.empty_cache()
    except Exception as e:
        log("清理异常（忽略）：%s" % type(e).__name__)
    ok = worst <= 1e-6
    log("结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
