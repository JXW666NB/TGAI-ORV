# -*- coding: utf-8 -*-
"""用 **Muon**（+ 分层即时更新）在 V100 32GB 上跑通 V2.7 全参 3B MoE 一步。

为什么 Muon 是关键：它每个 2D 参数只需要**一份**动量（AdamW 要两份）。
  fp32 动量：2.97B -> 11.1 GiB
  fp16 动量：2.97B ->  5.5 GiB   ← 再省一半

显存预算：
  权重 11.39 + Muon(fp16) 5.5 + 单层梯度 ~0.4 = 17.3 GiB ⇒ 留 14.5 GiB 给激活。

用法：python muon_3b.py [fp32|fp16]      # 动量精度，默认 fp16
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


def main(mode="fp16"):
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

    def sample(tag):
        f = rt.free_device_bytes() / GIB
        low[0] = min(low[0], f)
        log("  [%-12s] 空闲 %.2f GiB（峰值 %.2f GiB）" % (tag, f, total - low[0]))
        return f

    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    tr.attach_head(st)
    rt.empty_cache()
    sample("装载完")

    all_params, groups = [], {}
    for i in range(NL):
        ps = tr.params_of(i)
        groups[i] = ps
        all_params.extend(ps)
    all_params.extend(list(tr.head_params()))
    n_par = sum(int(p.numel()) for p in all_params)
    log("参数 %d 张量 / %.3f B" % (len(all_params), n_par / 1e9))

    try:
        if mode.startswith("adamw"):
            from tgai_go.train.opt_fp16mv import Fp16MomAdamW
            opt = Fp16MomAdamW(all_params, lr=2e-4, weight_decay=0.1,
                               fp32_state=(mode == "adamw-fp32"))
        else:
            opt = T.optim.Muon(all_params, lr=2e-4, momentum=0.95, weight_decay=0.0,
                               fp16_mom=(mode != "muon-fp32"))
        log("Muon 状态大小（按形状算）：%.2f GiB（每 2D 参数 %.0f 字节）"
            % (opt.state_bytes() / GIB, opt.bytes_per_param()))
        opt.set_layer_groups(groups)
        log("已登记 %d 个组（含 -1）" % len(opt.all_groups()))
        sample("优化器建好")

        rng = np.random.default_rng(0)
        ids = rng.integers(0, 30000, SEQ + 1, dtype=np.int64)
        j = np.arange(0, DH, 2, dtype=np.float32) / DH
        ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
        cos = T.tensor(np.cos(ang).astype(np.float32))
        sin = T.tensor(np.sin(ang).astype(np.float32))

        ys, xs = [], []
        x = T.tensor(rng.normal(0, 1, (SEQ, D)).astype(np.float32))
        for i in range(NL):
            ya = tr.fwd_attn(x, i, cos, sin)
            ys.append(np.asarray(ya.to_numpy(), np.float32))
            y = tr.moe_part(ya, i)
            xs.append(np.asarray(y.to_numpy(), np.float32))
            x = T.tensor(xs[-1])
            if i % 7 == 6:
                sample("前向%d/%d" % (i + 1, NL))

        gw = rng.normal(0, 0.01, (SEQ, D)).astype(np.float32)
        for i in range(NL - 1, -1, -1):
            y_t = T.tensor(xs[i], requires_grad=True)
            z = tr.moe_part(y_t, i)
            (z * T.tensor(gw)).sum().backward()
            xi = T.tensor(ys[i], requires_grad=True)
            ya = tr.fwd_attn(xi, i, cos, sin)
            (ya * T.tensor(gw)).sum().backward()
            opt.step_layers([i], t=1, release=True)
            if i % 7 == 0:
                sample("反向%d" % (NL - i))
        # 收尾：embed / norm / head 那一组
        for p in tr.head_params():
            pass
        opt.step_layers([-1], t=1, release=True)
        rt.sync()
        sample("完成")
        log("★★ 结论：3B 全参 Muon(%s) **跑通**，峰值 %.2f GiB，空闲还剩 %.2f GiB"
            % (mode, total - low[0], low[0]))
    except Exception as exc:
        log("结论：失败：%s: %s" % (type(exc).__name__, str(exc)[:200]))
    finally:
        try:
            st.close(); rt.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fp16")
