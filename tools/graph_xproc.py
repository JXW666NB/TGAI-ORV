# -*- coding: utf-8 -*-
"""分进程验证图重放等价性（消除同一进程内的交叉污染）。

用法（三次运行，互不共享状态）：
    python graph_xproc.py --mode eager  -> 写 grads_eager.npz
    python graph_xproc.py --mode graph  -> 写 grads_graph.npz
    python graph_xproc.py --mode cmp    -> 比较两者（判据 1e-6）

每个进程都从零构造 trainer、置零梯度、只用一次前反向，因此不存在"上一轮残留"。
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"
SEQ = 128
HERE = os.path.dirname(os.path.abspath(__file__))
F_EAGER = os.path.join(HERE, "grads_eager.npz")
F_GRAPH = os.path.join(HERE, "grads_graph.npz")


def build():
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    D, NH = int(h["d_model"]), int(h["n_heads"])
    cfg = dict(d_model=D, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]),
               rope_theta=float(h["rope_theta"]))
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    return T, st, tr, D, NH


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("eager", "graph", "cmp"), required=True)
    args = ap.parse_args()

    if args.mode == "cmp":
        a = np.load(F_EAGER)
        b = np.load(F_GRAPH)
        worst = 0.0
        for k in a.files:
            r = float(np.abs(a[k] - b[k]).max() / max(1e-9, float(np.abs(a[k]).max())))
            worst = max(worst, r)
            log("  %s  相对差 %.2e  max|eager|=%.4g max|graph|=%.4g"
                % (k, r, float(np.abs(a[k]).max()), float(np.abs(b[k]).max())))
        log("分进程等价性：最差 %.2e（判据 1e-6）⇒ %s"
            % (worst, "PASS" if worst <= 1e-6 else "FAIL"))
        return 0 if worst <= 1e-6 else 1

    T, st, tr, D, NH = build()
    rt = T._cuda()
    L = tr.layers[0]
    DH = D // NH
    THETA = 0.0
    from tgai_go.read_tg import close_tg, read_tg
    info = read_tg(MODEL_TG)
    THETA = float(info["header"]["rope_theta"])
    close_tg(info)

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

    def zero_grads():
        for k in ("q", "k", "v", "o"):
            for m in L[k]:
                shp = tuple(int(x) for x in m.weight.shape)
                m.weight.grad = T.tensor(np.zeros(shp, np.float32))

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

    def grads():
        out = {}
        for key, axis in (("q", 0), ("k", 0), ("v", 0), ("o", 1)):
            gs = [np.asarray(m.weight.grad.to_numpy(), np.float32) for m in L[key]]
            out[key] = np.concatenate(gs, axis=axis) if len(gs) > 1 else gs[0]
        return out

    zero_grads()
    if args.mode == "graph":
        rt.reserve_capture_arena(2 * 1024 ** 3)
        if not (rt.graph_arm() and rt.graph_begin()):
            log("捕获不可用"); return 2
        step()
        if not rt.graph_end():
            log("graph_end 失败"); return 2
        rt.graph_launch()                 # 重放才是图的结果（捕获期不保证写入）
        rt.sync()
        log("graph_stats: %s" % (rt.graph_stats(),))
        dst = F_GRAPH
    else:
        step()
        rt.sync()
        dst = F_EAGER

    g = grads()
    np.savez(dst, **g)
    log("已保存 %s：%s" % (os.path.basename(dst),
                          {k: (v.shape, float(np.abs(v).max())) for k, v in g.items()}))
    try:
        st.close(); rt.empty_cache()
    except Exception as e:
        log("清理异常（忽略）：%s" % type(e).__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
