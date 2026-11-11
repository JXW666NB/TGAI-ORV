# -*- coding: utf-8 -*-
"""3B 全参整步**耗时**实测（决定三星还是四星）。

配置：V2.7 28 层 2.97B，V100 32GB，Muon(fp16 动量) + 逐层即时更新，纯显存。

对照基准：PyTorch 的 Adafactor 步 **0.911 s**（这是判据里写的目标）。

用法：python step_time_3b.py [muon-fp16|adamw-fp16] [rep]
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

try:
    _spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
    _RUN = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_RUN)
    log = _RUN.log
except Exception:
    import datetime

    def log(msg):
        print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg), flush=True)

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"
SEQ, NL = 512, 28
GIB = float(1 << 30)


def main(mode="muon-fp16", rep=3):
    import torch
    import tgai_go.train as T
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.v27_train import TGState, V27Trainer

    info = read_tg(MODEL_TG); h = info["header"]; close_tg(info)
    D, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = D // NH, float(h["rope_theta"])
    cfg = dict(d_model=D, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    rt = T._cuda()
    total = 31.86
    low = [total]

    def peak():
        f = rt.free_device_bytes() / GIB
        low[0] = min(low[0], f)
        return f

    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    tr.attach_head(st)
    rt.empty_cache()
    peak()

    all_params, groups = [], {}
    for i in range(NL):
        ps = tr.params_of(i)
        groups[i] = ps
        all_params.extend(ps)
    all_params.extend(list(tr.head_params()))

    if mode.startswith("adamw"):
        from tgai_go.train.opt_fp16mv import Fp16MomAdamW
        opt = Fp16MomAdamW(all_params, lr=2e-4, weight_decay=0.1)
    else:
        # 注：`batch_ns` 已删除 —— 按形状分组批量做 Newton-Schulz 实测零收益
        # （35018 ms vs 34993 ms），说明瓶颈从不是启动次数。真正的瓶颈是
        # `sumsq_sequential`（单线程归约，101 ms/矩阵），0.9.49 已改并行。
        opt = T.optim.Muon(all_params, lr=2e-4, momentum=0.95, weight_decay=0.0,
                           fp16_mom=True, ns_fast=(mode != "muon-fp32"))
    opt.set_layer_groups(groups)
    log("%s 状态 %.2f GiB；组 %d 个" % (mode, opt.state_bytes() / GIB, len(opt.all_groups())))
    peak()

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30000, SEQ + 1, dtype=np.int64)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32))
    sin = T.tensor(np.sin(ang).astype(np.float32))
    gw = rng.normal(0, 0.01, (SEQ, D)).astype(np.float32)
    x0 = rng.normal(0, 1, (SEQ, D)).astype(np.float32)

    def one_step(t):
        # 前向链（逐层落主机，不保留图）
        ys, xs = [], []
        x = T.tensor(x0)
        for i in range(NL):
            ya = tr.fwd_attn(x, i, cos, sin)
            ys.append(np.asarray(ya.to_numpy(), np.float32))
            y = tr.moe_part(ya, i)
            xs.append(np.asarray(y.to_numpy(), np.float32))
            x = T.tensor(xs[-1])
        # 逐层反向 + 即时更新
        gwt = T.tensor(gw)
        for i in range(NL - 1, -1, -1):
            y_t = T.tensor(xs[i], requires_grad=True)
            z = tr.moe_part(y_t, i)
            (z * gwt).sum().backward()
            xi = T.tensor(ys[i], requires_grad=True)
            ya = tr.fwd_attn(xi, i, cos, sin)
            (ya * gwt).sum().backward()
            opt.step_layers([i], t=t, release=True)
        opt.step_layers([-1], t=t, release=True)

    # 预热一次（含各类 cuMemAlloc / 内核首次编译）
    one_step(1)
    rt.sync()
    log("预热完成，开始计时（%d 次）" % rep)
    peak()
    times = []
    for k in range(rep):
        rt.sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        one_step(k + 2)
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1))
        peak()
    mn, mean = min(times), sum(times) / len(times)
    log("3B 整步（%s，seq=%d，28 层）：min %.0f ms / mean %.0f ms" % (mode, SEQ, mn, mean))
    log("峰值显存 %.2f GiB（剩 %.2f GiB）" % (total - low[0], low[0]))
    log("对照 PyT Adafactor 步 911 ms ⇒ ORV/基准 = %.2fx %s"
        % (mn / 911.0, "（更慢）" if mn > 911 else "（更快）"))
    try:
        st.close(); rt.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "muon-fp16"
    r = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(a, r)
