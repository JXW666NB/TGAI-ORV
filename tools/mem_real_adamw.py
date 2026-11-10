# -*- coding: utf-8 -*-
"""实测 V2.7 **全参**（3.03B）在 V100 32GB 上跑一整步训练的真实显存。

用法（**一次只跑一种配置**，避免同进程互相污染）：
    python mem_real_adamw.py fp32     # 等价 PyTorch AdamW（m/v 全 fp32）
    python mem_real_adamw.py fp16     # ORV Fp16MomAdamW（m/v 存 fp16）

测的是：把模型**全部**参数（注意力 + MoE + 词表头）放进优化器，
跑「前向链 → MoE 反向 → 注意力反向 → 优化器一步」，阶段性采样空闲显存取最小值。
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
SEQ, NL = 512, 28
GIB = float(1 << 30)


def main(mode):
    import torch
    import tgai_go.train as T
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW
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
        log("  [%s] 空闲 %.2f GiB（当前峰值占用 %.2f GiB）" % (tag, f, total - low[0]))
        return f

    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    tr.attach_head(st)
    rt.empty_cache()
    sample("装载完")
    all_params = []
    for i in range(NL):
        all_params.extend(tr.params_of(i))
    all_params.extend(list(tr.head_params()))
    n_par = sum(int(p.numel()) for p in all_params)
    log("配置=%s；共 %d 个张量 / %.3f B 参数" % (mode, len(all_params), n_par / 1e9))

    try:
        opt = Fp16MomAdamW(all_params, lr=1e-4, weight_decay=0.1,
                           fp32_state=(mode == "fp32"))
        sample("优化器状态建好")

        rng = np.random.default_rng(0)
        ids = rng.integers(0, 30000, SEQ + 1, dtype=np.int64)
        j = np.arange(0, DH, 2, dtype=np.float32) / DH
        ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
        cos = T.tensor(np.cos(ang).astype(np.float32))
        sin = T.tensor(np.sin(ang).astype(np.float32))

        x = T.tensor(rng.normal(0, 1, (SEQ, D)).astype(np.float32))
        for i in range(NL):
            ya = tr.fwd_attn(x, i, cos, sin)
            x = tr.moe_part(ya, i)
            if i % 7 == 6:
                sample("前向 %d/%d" % (i + 1, NL))
        tr.loss_and_head_backward.__self__ if False else None
        gw = T.tensor(rng.normal(0, 0.01, (SEQ, D)).astype(np.float32))
        (x * gw).sum().backward()
        sample("反向完")
        opt.step()
        rt.sync()
        sample("优化器步进完")
        log("结论：%s 峰值占用约 **%.2f GiB** ⇒ 在 31.86 GiB 上%s"
            % (mode, total - low[0], "跑得通 ✔" if low[0] > 0 else "OOM ✘"))
    except Exception as exc:
        log("结论：%s 失败：%s: %s" % (mode, type(exc).__name__, str(exc)[:160]))
    finally:
        try:
            st.close(); rt.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fp16")
