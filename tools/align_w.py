# -*- coding: utf-8 -*-
"""定案：ORV 的 MoE 权重到底是"只在选中 k 上 softmax"还是"全部专家 softmax 再取"。

已知：算子级（gate/up/silu/down）全对，但整套 MoE 在 |Y|大时差 4.6e-2；
且 |g|max 达到 88.68（fp32 exp 溢出边界）。

做法：numpy 造两个参考实现
    ref_topk : w = softmax(z[k])            （据称是产线/CPU 语义）
    ref_all  : w = softmax(z) 再取 k 个      （CUDA 分支代码看起来是这样）
分别与 ORV 的输出比，看匹配哪个、以及 z 的极值是否越过 exp 溢出边界。

运行：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_w.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
MODEL_PT = r"D:\TGAI\models\V2.7\tgai_v27_v5_merged.pt"
SEQ, D = 64, 1024


def relerr(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.v27_train import V27Trainer, TGState
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    cfg = dict(d_model=int(h["d_model"]), n_heads=int(h["n_heads"]), d_ff=int(h["d_ff"]),
               n_experts=int(h["n_experts"]), n_activated=int(h["n_activated"]),
               n_layers=int(h["n_layers"]), rope_theta=float(h["rope_theta"]))
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    moe = tr.layers[0]["moe"]
    sd = torch.load(MODEL_PT, map_location="cpu", mmap=True, weights_only=False)["model_state_dict"]
    p0 = "blocks.0.moe."
    Wr = np.asarray(sd[p0 + "router.weight"].float().numpy(), np.float32)
    bq = np.asarray(sd[p0 + "_qb_bias"].float().numpy(), np.float32) if (p0 + "_qb_bias") in sd else None
    E = int(h["n_experts"]); K = int(h["n_activated"])
    Wg = [np.asarray(sd[p0 + "experts.%d.w1.weight" % e].float().numpy(), np.float32) for e in range(E)]
    Wu = [np.asarray(sd[p0 + "experts.%d.w2.weight" % e].float().numpy(), np.float32) for e in range(E)]
    Wd = [np.asarray(sd[p0 + "experts.%d.w3.weight" % e].float().numpy(), np.float32) for e in range(E)]

    base = np.random.default_rng(0).normal(0, 1, (SEQ, D)).astype(np.float32)

    def refs(Y):
        z = Y @ Wr.T + (bq if bq is not None else 0.0)
        idx = np.argsort(-z, axis=-1)[:, :K]
        sel = np.take_along_axis(z, idx, axis=-1)
        # A: 只在选中 k 上 softmax
        eA = np.exp(sel - sel.max(axis=-1, keepdims=True))
        wA = eA / eA.sum(axis=-1, keepdims=True)
        # B: 全部专家 softmax 再取（无 max 减法，模拟可能的内核写法）
        eB = np.exp(z)
        wB = np.take_along_axis(eB / eB.sum(axis=-1, keepdims=True), idx, axis=-1)
        # C: 全部专家 softmax（带 max 减法，数值安全）
        zs = z - z.max(axis=-1, keepdims=True)
        eC = np.exp(zs)
        wC = np.take_along_axis(eC / eC.sum(axis=-1, keepdims=True), idx, axis=-1)

        def combine(w):
            out = np.zeros_like(Y)
            for k in range(K):
                for e in range(E):
                    m = idx[:, k] == e
                    if not m.any():
                        continue
                    xe = Y[m]
                    g = xe @ Wg[e].T
                    u = xe @ Wu[e].T
                    hh = (g / (1.0 + np.exp(-g))) * u
                    out[m] += w[m, k:k + 1] * (hh @ Wd[e].T)
            return out
        return z, wA, wB, wC, combine(wA), combine(wB), combine(wC)

    for scale in (1.0, 4.0, 8.0):
        Y = (base * scale).astype(np.float32)
        o = np.asarray(moe(T.tensor(Y)).to_numpy(), np.float32)
        z, wA, wB, wC, rA, rB, rC = refs(Y)
        log("scale=%-4g |z|max=%10.2f  exp(z)含inf=%s" % (scale, float(np.abs(z).max()), bool(np.isinf(np.exp(z.astype(np.float64))).any())))
        log("   ORV vs A(仅选中softmax) = %.2e" % relerr(o, rA))
        log("   ORV vs B(全部softmax)   = %.2e" % relerr(o, rB))
        log("   ORV vs C(全部+max减法)  = %.2e" % relerr(o, rC))
        log("   w 对照: A 行和=%.4f  B 最大权重=%.4f  首行 A=%s B=%s"
            % (float(wA.sum(-1).mean()), float(wB.max()),
               np.round(wA[0], 4), np.round(wB[0], 4)))
    st.close()


if __name__ == "__main__":
    main()
