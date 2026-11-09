# -*- coding: utf-8 -*-
"""整步集成：前向 eager（fwd_attn + moe_part）、反向逐层图重放（MoE 反传仍 eager）。

判据：loss 与参数梯度逐位相同；CUDA event 计时给出整步时间。
运行（需独占 GPU）：python graph_whole_step.py
"""
import ctypes
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
    x0 = np.ascontiguousarray(np.asarray(st["token_embedding.weight"][ids[:-1]], np.float32))

    def zero_all():
        for i in range(NL):
            for k in ("q", "k", "v", "o"):
                for m in tr.layers[i][k]:
                    shp = tuple(int(x) for x in m.weight.shape)
                    m.weight.grad = T.tensor(np.zeros(shp, np.float32))
        for p in tr.head_params():
            p.grad = None

    def fwd_chain():
        """返回 (ys_attn, xs)；前向 arithmetic 与 tr.fwd 相同。"""
        ys_attn, xs = [], [x0]
        x = T.tensor(x0)
        for i in range(NL):
            ya = tr.fwd_attn(x, i, cos, sin)
            ys_attn.append(np.asarray(ya.to_numpy(), np.float32))
            y = tr.moe_part(ya, i)
            xn = np.asarray(y.to_numpy(), np.float32)
            xs.append(xn)
            x = T.tensor(xn)
        return ys_attn, xs

    def eager_step():
        zero_all()
        ys_attn, xs = fwd_chain()
        loss, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
        gg = np.asarray(g, np.float32)
        for i in range(NL - 1, -1, -1):
            xi = T.tensor(xs[i], requires_grad=True)
            ya = tr.fwd_attn(xi, i, cos, sin)
            z = tr.moe_part(ya, i)
            (z * T.tensor(gg)).sum().backward()
            gg = np.asarray(xi.grad.to_numpy(), np.float32)
        return float(np.asarray(loss.to_numpy()))

    def gsum_all():
        tot = 0.0
        for i in range(NL):
            for k in ("q", "k", "v", "o"):
                for m in tr.layers[i][k]:
                    tot += float(np.abs(np.asarray(m.weight.grad.to_numpy(), np.float32)).sum())
        return tot

    torch.cuda.synchronize(); t0 = time.time()
    loss_e = eager_step()
    torch.cuda.synchronize(); t_eager = time.time() - t0
    g_e = gsum_all()
    log("eager 整步（前向+loss+反向）：%.1f ms  loss=%.8f  梯度合计=%.6f"
        % (t_eager * 1000, loss_e, g_e))

    # 捕获每层图（fwd_attn + 反向），输入/梯度缓冲固定，保留 x 引用以读梯度
    ys_attn, xs = fwd_chain()
    x_bufs = [rt.h2d(xs[i]) for i in range(NL)]
    gw_bufs = [rt.h2d(np.zeros((SEQ, D), np.float32)) for _ in range(NL)]
    x_refs = []
    grad_refs = []          # 捕获期的 .grad 对象（图写进这些缓冲，读它们才对）
    rt.reserve_capture_arena(ARENA)
    for i in range(NL):
        for k in ("q", "k", "v", "o"):
            for m in tr.layers[i][k]:
                shp = tuple(int(x) for x in m.weight.shape)
                m.weight.grad = T.tensor(np.zeros(shp, np.float32))
        if not (rt.graph_arm() and rt.graph_begin()):
            log("第 %d 层 graph_begin 失败" % i); return 2
        xi = T.Tensor._from_gpu(x_bufs[i], requires_grad=True)
        ya = tr.fwd_attn(xi, i, cos, sin)
        (ya * T.Tensor._from_gpu(gw_bufs[i])).sum().backward()
        if not rt.graph_end():
            log("第 %d 层 graph_end 失败" % i); return 2
        x_refs.append(xi)
        grad_refs.append([m.weight.grad for k in ("q", "k", "v", "o") for m in tr.layers[i][k]])
    log("已捕获 %d 张图" % rt.graph_count())

    def h2d_into(buf, arr):
        a = np.ascontiguousarray(arr, np.float32)
        rt._activate()
        rt._chk(rt.cu.cuMemcpyHtoD(ctypes.c_uint64(int(buf.ptr)),
                                   a.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_size_t(a.nbytes)), "cuMemcpyHtoD")

    def graph_step():
        # 注意：**不能 zero_all()** —— 那会替换 .grad 对象，图仍写捕获期缓冲。
        # 只清 head 的梯度（它走 eager）。
        for p in tr.head_params():
            p.grad = None
        ys2, xs2 = fwd_chain()
        loss, g = tr.loss_and_head_backward(xs2[-1], ids[1:], cos, sin)
        gg = np.asarray(g, np.float32)
        for i in range(NL - 1, -1, -1):
            y_t = T.tensor(ys2[i], requires_grad=True)      # MoE 反传（eager），拿 dL/dy_attn
            z = tr.moe_part(y_t, i)
            (z * T.tensor(gg)).sum().backward()
            h2d_into(gw_bufs[i], np.asarray(y_t.grad.to_numpy(), np.float32))
            rt.graph_launch_slot(i)                         # attention 反传（图）
            gg = np.asarray(x_refs[i].grad.to_numpy(), np.float32)
        return float(np.asarray(loss.to_numpy()))

    torch.cuda.synchronize(); t0 = time.time()
    loss_g = graph_step()
    torch.cuda.synchronize(); t_graph = time.time() - t0
    g_g = sum(float(np.abs(np.asarray(t.to_numpy(), np.float32)).sum())
              for layer in grad_refs for t in layer)
    log("graph 整步（前向+loss+反向）：%.1f ms  loss=%.8f  梯度合计=%.6f"
        % (t_graph * 1000, loss_g, g_g))
    log("loss 相对差 %.2e；梯度合计相对差 %.2e"
        % (abs(loss_g - loss_e) / max(1e-9, abs(loss_e)),
           abs(g_g - g_e) / max(1e-9, abs(g_e))))
    log("整步：%.1f ms → %.1f ms（%.2fx，省 %.0f ms）"
        % (t_eager * 1000, t_graph * 1000, t_eager / max(t_graph, 1e-9),
           (t_eager - t_graph) * 1000))
    try:
        rt.graph_release_all(); st.close(); rt.empty_cache()
    except Exception as e:
        log("清理异常（忽略）：%s" % type(e).__name__)
    ok = abs(g_g - g_e) / max(1e-9, abs(g_e)) <= 1e-6
    log("结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
