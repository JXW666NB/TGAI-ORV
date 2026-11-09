# -*- coding: utf-8 -*-
"""对齐定位 ⑥：MoE 误差与**输入幅值**的关系 + 单专家逐算子核对。

已知（上一轮）：
    * 同一输入 h2(y) 下，ORV MoE 与 torch MoE 差 55%，而路由完全一致（64/64）；
    * 同一输入 h2(x0)（幅值更小）下，两者只差 7e-7。
⇒ 强烈指向 **ORV 的 MoE 在幅值较大时数值算错**（而非路由）。

本脚本：
  S1 把 h2 按 0.1/1/2/4/8 倍缩放，测两侧 MoE 相对误差随幅值的走势；
  S2 单专家核对：numpy 手算 silu(x@gate.T)*(x@up.T)@down.T，与 ORV 的 expert 模块、torch 的 moe 分支逐一对齐。

运行：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_moe3.py
"""
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")
sys.path.insert(0, r"D:\tgai_orv")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
MODEL_PT = r"D:\TGAI\models\V2.7\tgai_v27_v5_merged.pt"
SEQ = 64


def relerr(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def main():
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    DM = int(h["d_model"]); NE = int(h["n_experts"]); NA = int(h["n_activated"])
    cfg = dict(d_model=DM, n_heads=int(h["n_heads"]), d_ff=int(h["d_ff"]),
               n_experts=NE, n_activated=NA, n_layers=int(h["n_layers"]),
               rope_theta=float(h["rope_theta"]))
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)

    sd = torch.load(MODEL_PT, map_location="cpu", mmap=True, weights_only=False)["model_state_dict"]
    p0 = "blocks.0.moe."
    W = {k: np.asarray(sd[p0 + k].float().numpy(), np.float32) for k in
         ("router.weight", "_qb_bias") if (p0 + k) in sd}
    for k in ("router.weight",):
        W.setdefault(k, np.asarray(sd[p0 + k].float().numpy(), np.float32))

    # 取一个真实量级的输入：直接构造（避免再跑整块），用 embedding 的行均值×若干倍
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (SEQ, DM)).astype(np.float32)

    def moe_orv(Y):
        return np.asarray(tr.layers[0]["moe"](T.tensor(Y)).to_numpy(), np.float32)

    # numpy 参考实现（按文档语义：z=xW^T+b → topk2 → softmax(选中2) → SwiGLU 加权和）
    def moe_np(Y):
        z = Y @ W["router.weight"].T
        if "_qb_bias" in W:
            z = z + W["_qb_bias"]
        idx = np.argsort(-z, axis=-1)[:, :NA]
        sel = np.take_along_axis(z, idx, axis=-1)
        e = np.exp(sel - sel.max(axis=-1, keepdims=True))
        w = e / e.sum(axis=-1, keepdims=True)
        out = np.zeros_like(Y)
        for k in range(NA):
            for eid in range(NE):
                m = idx[:, k] == eid
                if not m.any():
                    continue
                xe = Y[m]
                g = xe @ np.asarray(sd[p0 + "experts.%d.w1.weight" % eid].float().numpy(), np.float32).T
                u = xe @ np.asarray(sd[p0 + "experts.%d.w2.weight" % eid].float().numpy(), np.float32).T
                hh = (g / (1.0 + np.exp(-g))) * u
                d = hh @ np.asarray(sd[p0 + "experts.%d.w3.weight" % eid].float().numpy(), np.float32).T
                out[m] += w[m, k:k + 1] * d
        return out

    log("=== S1 幅值扫描（ORV vs numpy 参考）===")
    for s in (0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0):
        Y = (base * s).astype(np.float32)
        log("   scale=%-5g |Y|max=%8.2f  ORV vs numpy 相对误差=%.2e"
            % (s, float(np.abs(Y).max()), relerr(moe_orv(Y), moe_np(Y))))

    log("=== S2 单专家逐算子（scale=2）===")
    Y = (base * 2.0).astype(np.float32)
    eid = 0
    Wo = lambda k: np.asarray(sd[p0 + "experts.%d.%s.weight" % (eid, k)].float().numpy(), np.float32)
    expert = tr.layers[0]["moe"].experts[eid]
    g_np = Y @ Wo("w1").T
    u_np = Y @ Wo("w2").T
    h_np = (g_np / (1.0 + np.exp(-g_np))) * u_np
    d_np = h_np @ Wo("w3").T
    gate_o = np.asarray(expert.gate(T.tensor(Y)).to_numpy(), np.float32)
    up_o = np.asarray(expert.up(T.tensor(Y)).to_numpy(), np.float32)
    log("  gate(x) 相对误差=%.2e" % relerr(gate_o, g_np))
    log("  up(x)   相对误差=%.2e" % relerr(up_o, u_np))
    try:
        act = getattr(expert, "act", None)
        log("  expert 属性: %s" % [a for a in dir(expert) if not a.startswith('_')][:10])
    except Exception:
        pass
    st.close()


if __name__ == "__main__":
    main()
