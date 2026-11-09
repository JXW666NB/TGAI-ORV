# -*- coding: utf-8 -*-
"""fused 路径 NaN 的决定性诊断（需先 `python tools/apply_fused_heads.py`）。

依次检查：
  1. load_layer 之后 q/o 的权重是否有限（默认 vs fused）—— 判定"懒分配未物化"
  2. 先用模块调用物化一次，再切片，看 q_all 是否有限
  3. 打印 fused 前向各阶段（投影 / rope / 切片 / softmax 后）的 max|.| 与是否有限

运行（需独占 GPU）：python fused_diag.py
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
SEQ = 8


def fin(tag, t):
    a = np.asarray(t.to_numpy() if hasattr(t, "to_numpy") else t, np.float32)
    ok = bool(np.isfinite(a).all())
    log("  %-26s shape=%-14s finite=%-5s max|.|=%.6g" % (tag, str(a.shape), ok, float(np.abs(a).max())))
    return ok


def main():
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

    def make(fused):
        st = TGState(MODEL_TG)
        tr = V27Trainer(cfg, n_layers=1)
        tr.fused_heads = bool(fused)
        tr.load_all(st)
        return tr, st

    for fused in (False, True):
        log("=== fused_heads=%s ===" % fused)
        tr, st = make(fused)
        L = tr.layers[0]
        okq = fin("q[0].weight", L["q"][0].weight)
        oko = fin("o[0].weight", L["o"][0].weight)
        if not (okq and oko):
            log("  !! 权重非有限 —— 懒分配未物化，这是 NaN 的直接来源")
        # 用一次模块调用，再切片
        x = T.tensor(np.random.default_rng(0).normal(0, 1, (SEQ, D)).astype(np.float32))
        hh = L["ln1"](x)
        q_full = L["q"][0](hh)
        fin("q[0](h)", q_full)
        try:
            fin("q_full[:, 0:DH]", q_full[:, 0:8])
        except Exception as e:
            log("  切片失败：%s: %s" % (type(e).__name__, str(e)[:80]))
        if fused and L["q"][0].weight.shape[0] == D:
            sl = slice(0, 8)
            try:
                fin("weight[:, sl].T", L["o"][0].weight[:, sl].T)
            except Exception as e:
                log("  o 切片失败：%s: %s" % (type(e).__name__, str(e)[:80]))
        st.close()
        T._cuda().empty_cache()


if __name__ == "__main__":
    main()
