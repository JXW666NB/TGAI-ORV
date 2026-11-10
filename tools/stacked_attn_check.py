# -*- coding: utf-8 -*-
"""`fwd_attn_stacked`（堆叠 head）与 `fwd_attn`（逐头循环）的等价性 + 速度。

判据：前向与**梯度**都要一致（rope/softmax 的求和顺序不同，允许 ≤1e-5 相对差）。
速度：同一随机输入上各跑 N 次，报 ms 与加速比。

运行（需独占 GPU，但 CPU 也能跑通数值部分）：python stacked_attn_check.py
"""
import os
import sys
import time as _time

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "python"))

# 日志函数：GPU 机上用 v27_train_run.py 的 log（带时间戳），本地没有就退回 print
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
    _RUN = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_RUN)
    log = _RUN.log
except Exception:
    import datetime

    def log(msg):
        print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg), flush=True)

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"
NH_LAYERS = 2


def rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.v27_train import TGState, V27Trainer

    # 先定设备（下面的 seq 与分支都要用它）
    use_cuda = True
    try:
        T.set_default_device("cuda")
        T._cuda().free_device_bytes()
    except Exception:
        T.set_default_device("cpu")
        use_cuda = False
    log("设备: %s" % ("cuda" if use_cuda else "cpu"))

    # 优先用真实模型；本地没有 .tg 时用合成小配置（随机权重）做数值对照
    if os.path.exists(MODEL_TG):
        info = read_tg(MODEL_TG); h = info["header"]; close_tg(info)
        D, NH = int(h["d_model"]), int(h["n_heads"])
        DH, THETA = D // NH, float(h["rope_theta"])
        cfg = dict(d_model=D, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
                   n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]),
                   rope_theta=THETA)
        st = TGState(MODEL_TG)
        tr = V27Trainer(cfg, n_layers=NH_LAYERS)
        tr.load_all(st)
    else:
        log("本地没有 %s，改用合成分层（数值对照同样有效）" % MODEL_TG)
        D, NH, THETA = 64, 4, 1e4
        DH = D // NH
        cfg = dict(d_model=D, n_heads=NH, d_ff=128, n_experts=4, n_activated=2,
                   n_layers=NH_LAYERS, rope_theta=THETA)
        tr = V27Trainer(cfg, n_layers=NH_LAYERS)
        # 逐层建 + 随机初始化（模拟 load_all 之后的形状）
        rngw = np.random.default_rng(7)
        for _ in range(NH_LAYERS):
            L = tr._build_layer()
            for key in ("ln1", "ln2"):
                if hasattr(L[key], "weight"):
                    L[key].weight = T.nn.Parameter(np.ones(D, np.float32))
            for key in ("q", "k", "v"):
                for m in L[key]:
                    m.weight = T.nn.Parameter((rngw.normal(0, 0.08, (DH, D))).astype(np.float32))
            for m in L["o"]:
                m.weight = T.nn.Parameter((rngw.normal(0, 0.08, (D, DH))).astype(np.float32))
            tr.layers.append(L)
        st = None

    seq = 64 if not use_cuda else 256
    rng = np.random.default_rng(0)
    x_np = rng.normal(0, 1, (seq, D)).astype(np.float32)
    gw = rng.normal(0, 1, (seq, D)).astype(np.float32)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(seq, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))

    def zero(i):
        for k in ("q", "k", "v", "o"):
            for m in tr.layers[i][k]:
                m.weight.grad = None

    def _np(x):
        """CPU 上 .grad 是 numpy，CUDA 上是 Tensor —— 统一取成 numpy。"""
        return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)

    def run(fn, i):
        zero(i)
        xi = T.tensor(x_np, requires_grad=True)
        y = fn(xi, i, cos, sin)
        yn = _np(y)
        (y * T.tensor(gw)).sum().backward()
        gx = _np(xi.grad)
        gs = [_np(m.weight.grad)
              for k in ("q", "k", "v", "o") for m in tr.layers[i][k]]
        return yn, gx, gs

    ok = True
    for i in range(NH_LAYERS):
        # 显式对比"逐头循环"与"堆叠"两个实现（fwd_attn 默认已走堆叠，不能拿它当参照）
        y1, x1, g1 = run(tr.fwd_attn_loop, i)
        y2, x2, g2 = run(tr.fwd_attn_stacked, i)
        r = [rel(y2, y1), rel(x2, x1)] + [rel(b, a) for a, b in zip(g1, g2)]
        log("第 %d 层：前向 %.2e  dL/dx %.2e  权重梯度最大 %.2e"
            % (i, r[0], r[1], max(r[2:]) if len(r) > 2 else 0))
        ok = ok and max(r) <= 1e-5

    # 速度
    def bench(fn, rep=3):
        fn(T.tensor(x_np), 0, cos, sin)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = _time.time()
        for _ in range(rep):
            fn(T.tensor(x_np), 0, cos, sin)
        if use_cuda:
            torch.cuda.synchronize()
        return (_time.time() - t0) / rep * 1000.0

    t1 = bench(tr.fwd_attn_loop)
    t2 = bench(tr.fwd_attn_stacked)
    log("seq=%d 单层注意力前向：逐头 %.1f ms → 堆叠 %.1f ms（%.2fx）" % (seq, t1, t2, t1 / max(t2, 1e-9)))
    log("总结论：%s" % ("PASS" if ok else "FAIL"))
    try:
        if st is not None:
            st.close()
        if use_cuda:
            T._cuda().empty_cache()
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
