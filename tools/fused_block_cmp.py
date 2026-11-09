# -*- coding: utf-8 -*-
"""最小、串行的 fused vs 默认 attention 块对比（需先 apply_fused_heads.py）。

设计原则（针对之前 NaN 的教训）：
  * **同一时刻只有一个 trainer 活着**：先跑默认、拷出结果、close 并 empty_cache，再建 fused；
  * 只做一次前向，不做计时循环（计时另外测，避免把分配问题混进来）；
  * 全程用 to_numpy()，不用 .data。

运行（需独占 GPU）：python fused_block_cmp.py
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
SEQ = 32


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

    x0 = np.random.default_rng(0).normal(0, 1, (SEQ, D)).astype(np.float32)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos_np, sin_np = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    def run(fused):
        st = TGState(MODEL_TG)
        tr = V27Trainer(cfg, n_layers=1)
        tr.fused_heads = bool(fused)
        tr.load_all(st)
        L = tr.layers[0]
        if fused and int(L["q"][0].weight.shape[0]) != D:
            log("!! 补丁未应用（q[0] 形状 %s）" % (tuple(int(s) for s in L["q"][0].weight.shape),))
            st.close()
            return None
        cos = T.tensor(cos_np)
        sin = T.tensor(sin_np)
        x = T.tensor(x0)
        hh = L["ln1"](x)
        inv = 1.0 / np.sqrt(DH)
        mask = tr._causal_mask(SEQ)
        if fused:
            cos_t = T.tensor(np.tile(cos_np, (1, NH)))
            sin_t = T.tensor(np.tile(sin_np, (1, NH)))
            q_all = T._rope_composed(L["q"][0](hh), cos_t, sin_t)
            k_all = T._rope_composed(L["k"][0](hh), cos_t, sin_t)
            v_all = L["v"][0](hh)
            acc = None
            for i in range(NH):
                sl = slice(i * DH, (i + 1) * DH)
                pr = T.functional.softmax((q_all[:, sl] @ k_all[:, sl].T) * inv + mask, axis=-1)
                part = (pr @ v_all[:, sl]) @ L["o"][0].weight[:, sl].T
                acc = part if acc is None else acc + part
        else:
            acc = None
            for i in range(NH):
                q = T._rope_composed(L["q"][i](hh), cos, sin)
                k = T._rope_composed(L["k"][i](hh), cos, sin)
                v = L["v"][i](hh)
                pr = T.functional.softmax((q @ k.T) * inv + mask, axis=-1)
                part = L["o"][i](pr @ v)
                acc = part if acc is None else acc + part
        out = np.asarray(acc.to_numpy(), np.float32).copy()
        st.close()
        rt.empty_cache()
        torch.cuda.synchronize()
        return out

    a = run(False)
    if a is None:
        return 2
    log("默认：finite=%s max|.|=%.6g" % (bool(np.isfinite(a).all()), float(np.abs(a).max())))
    b = run(True)
    if b is None:
        return 2
    log("fused：finite=%s max|.|=%.6g" % (bool(np.isfinite(b).all()), float(np.abs(b).max())))
    if np.isfinite(a).all() and np.isfinite(b).all():
        rel = float(np.abs(a - b).max() / max(1e-9, float(np.abs(a).max())))
        log("相对差 = %.2e（判据 1e-6）⇒ %s" % (rel, "PASS" if rel <= 1e-6 else "FAIL"))
        return 0 if rel <= 1e-6 else 1
    log("存在非有限值 ⇒ 仍需定位")
    return 1


if __name__ == "__main__":
    sys.exit(main())
