# -*- coding: utf-8 -*-
"""A2 关键判断：把 attention 块（无 MoE、无 D2H）捕获成 CUDA Graph，与逐算子对比。

要回答：如果图捕获能省掉主机侧逐算子开销，那么这个块应当明显变快；
        若不变快，说明 A2 的价值也有限，不值得动 MoE 内核。

判据：输出与梯度 ≤1e-6；并给出耗时。

运行（需独占 GPU）：python graph_attn_test.py
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
SEQ = 512
REP = 5


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
    log("graph 可用: %s" % rt.graph_available())

    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    L = tr.layers[0]

    # 固定输入（图捕获要求地址稳定）
    x_np = np.random.default_rng(0).normal(0, 1, (SEQ, D)).astype(np.float32)
    gw_np = np.random.default_rng(1).normal(0, 1, (SEQ, D)).astype(np.float32)
    x_buf = rt.h2d(x_np)
    gw_buf = rt.h2d(gw_np)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32))
    sin = T.tensor(np.sin(ang).astype(np.float32))
    mask = tr._causal_mask(SEQ)
    inv = 1.0 / np.sqrt(DH)

    def clear_grads():
        for k in ("q", "k", "v", "o"):
            for m in L[k]:
                m.weight.grad = None

    def step():
        x = T.Tensor._from_gpu(x_buf, requires_grad=True)
        hh = L["ln1"](x)
        acc = None
        for i in range(NH):
            q = T._rope_composed(L["q"][i](hh), cos, sin)
            k = T._rope_composed(L["k"][i](hh), cos, sin)
            v = L["v"][i](hh)
            pr = T.functional.softmax((q @ k.T) * inv + mask, axis=-1)
            part = L["o"][i](pr @ v)
            acc = part if acc is None else acc + part
        (acc * T.Tensor._from_gpu(gw_buf)).sum().backward()
        return acc

    def grads_np():
        out = {}
        for key, axis in (("q", 0), ("k", 0), ("v", 0), ("o", 1)):
            gs = [np.asarray(m.weight.grad.to_numpy(), np.float32) for m in L[key]]
            out[key] = np.concatenate(gs, axis=axis) if len(gs) > 1 else gs[0]
        return out

    # 逐算子：结果 + 计时
    clear_grads(); y = step(); torch.cuda.synchronize()
    out_eager = np.asarray(y.to_numpy(), np.float32).copy()
    g_eager = grads_np()
    ptr_before = int(L["q"][0].weight.grad._gpu.ptr)
    tray = []
    for _ in range(REP):
        clear_grads(); torch.cuda.synchronize(); t0 = time.time()
        step(); torch.cuda.synchronize(); tray.append(time.time() - t0)
    t_eager = min(tray)

    # 图捕获：起始梯度必须是**确定值**（置零缓冲），不能是 None ——
    # 否则图烘焙的是捕获窗口内新建的缓冲，里面是垃圾（上一轮 1.62e+01 的来源）
    def zero_grads():
        for k in ("q", "k", "v", "o"):
            for m in L[k]:
                shp = tuple(int(x) for x in m.weight.shape)
                m.weight.grad = T.tensor(np.zeros(shp, np.float32))

    zero_grads()
    # 运行时要求捕获前预留 arena（否则池未命中就回退）
    if hasattr(rt, "reserve_capture_arena"):
        rt.reserve_capture_arena(2 * 1024 ** 3)
        log("已预留捕获 arena 2 GiB")
    ok = rt.graph_arm() and rt.graph_begin()
    log("graph_begin=%s" % ok)
    if not ok:
        st.close(); return 2
    try:
        step()
    except Exception as e:
        log("捕获期间失败：%s: %s" % (type(e).__name__, str(e)[:120]))
        rt.graph_abort(str(e)); st.close(); return 2
    if not rt.graph_end():
        log("graph_end 失败")
        st.close(); return 2

    # 捕获期的那次执行是真实的，用它做等价性对比（重放只测时间：
    # 重放前若清空 p.grad，Python 侧就没有对象可读，而图里烘焙的是捕获时的缓冲）
    g_cap = grads_np()
    log("graph_stats: %s" % (rt.graph_stats(),))
    ok_l = rt.graph_launch()
    log("graph_launch 返回: %s" % ok_l)
    torch.cuda.synchronize()
    tray = []
    for _ in range(REP):
        torch.cuda.synchronize(); t0 = time.time()
        rt.graph_launch(); torch.cuda.synchronize(); tray.append(time.time() - t0)
    t_graph = min(tray)
    g_graph = g_cap

    worst = 0.0
    for key in ("q", "k", "v", "o"):
        r = float(np.abs(g_eager[key] - g_graph[key]).max() /
                  max(1e-9, float(np.abs(g_eager[key]).max())))
        worst = max(worst, r)
    log("梯度（逐算子 vs 图重放）：最差 %.2e（判据 1e-6）" % worst)
    log("耗时：逐算子 %.2f ms → 图重放 %.2f ms（%.2fx）"
        % (t_eager * 1000, t_graph * 1000, t_eager / max(t_graph, 1e-9)))
    # 等价性：重放是**累加**到同一缓冲的，所以用"重放一次前后之差"与逐算子结果比较
    g0 = grads_np()
    rt.graph_launch(); torch.cuda.synchronize()
    g1 = grads_np()
    worst = 0.0
    for key in ("q", "k", "v", "o"):
        single = g1[key] - g0[key]
        r = float(np.abs(single - g_eager[key]).max() /
                  max(1e-9, float(np.abs(g_eager[key]).max())))
        worst = max(worst, r)
        log("  单次重放增量 vs 逐算子  %s：%.2e" % (key, r))
    log("单次重放等价性：最差 %.2e（判据 1e-6）" % worst)
    try:
        ptr_after = int(L["q"][0].weight.grad._gpu.ptr)
        log("q[0].grad 缓冲地址（重放后）= 0x%x，捕获前记录=0x%x" % (ptr_after, ptr_before))
    except Exception as e:
        log("取 grad 地址失败：%s" % type(e).__name__)
    # CUDA event 计时（不依赖 Python 时钟）
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    INNER = 10
    e0.record()
    for _ in range(INNER):
        rt.graph_launch()
    e1.record()
    e1.synchronize()
    log("event 计时：图重放 %.2f ms/次" % (e0.elapsed_time(e1) / INNER))
    e0.record()
    for _ in range(INNER):
        step()
    e1.record()
    e1.synchronize()
    log("event 计时：逐算子 %.2f ms/次" % (e0.elapsed_time(e1) / INNER))
    try:                                  # arena 存活时 empty_cache 会崩（运行时缺陷），不让它盖住结果
        st.close(); rt.empty_cache()
    except Exception as e:
        log("清理阶段异常（不影响上面的结论）：%s: %s" % (type(e).__name__, str(e)[:90]))
    log("结论：%s" % ("PASS（图可用且一致）" if worst <= 1e-6 else "FAIL（不一致）"))
    return 0 if worst <= 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
