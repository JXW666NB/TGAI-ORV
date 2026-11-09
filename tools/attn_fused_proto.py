# -*- coding: utf-8 -*-
"""真实权重下：attention 块 per-head vs 全宽投影（原型，**不改库代码**）。

只替换投影写法（rope 用全宽+平铺表、o 用全宽列切片），MoE 不参与。
输出三件事：块输出一致性、发射次数、耗时。

运行（需独占 GPU）：python attn_fused_proto.py
"""
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
SEQ = 128
REP = 5


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    DM, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = DM // NH, float(h["rope_theta"])
    cfg = dict(d_model=DM, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    L = tr.layers[0]
    rt = T._cuda()
    inv = 1.0 / np.sqrt(DH)

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30000, SEQ, dtype=np.int64)
    x0 = np.ascontiguousarray(np.asarray(st["token_embedding.weight"][ids], np.float32))
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos_np, sin_np = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    cos, sin = T.tensor(cos_np), T.tensor(sin_np)
    mask = tr._causal_mask(SEQ)

    def wname(k):
        # L[k] 是 Linear 模块列表（不是参数），取 .weight
        return np.concatenate([np.asarray(mod.weight.data, np.float32) for mod in L[k]],
                              axis=0 if k != "o" else 1)

    Wq, Wk, Wv, Wo = (T.tensor(wname(k)) for k in ("q", "k", "v", "o"))
    tcos = T.tensor(np.tile(cos_np, (1, NH)))
    tsin = T.tensor(np.tile(sin_np, (1, NH)))

    counts = Counter()
    orig = type(rt).launch

    def counted(self, name, *a, **kw):
        counts[name] += 1
        return orig(self, name, *a, **kw)

    type(rt).launch = counted

    def per_head(x):
        acc = None
        for i in range(NH):
            q = T._rope_composed(L["q"][i](x), cos, sin)
            k = T._rope_composed(L["k"][i](x), cos, sin)
            v = L["v"][i](x)
            pr = T.functional.softmax((q @ k.T) * inv + mask, axis=-1)
            part = L["o"][i](pr @ v)
            acc = part if acc is None else acc + part
        return acc

    def fused(x):
        q_all = T._rope_composed(x @ Wq.T, tcos, tsin)
        k_all = T._rope_composed(x @ Wk.T, tcos, tsin)
        v_all = x @ Wv.T
        acc = None
        for i in range(NH):
            sl = slice(i * DH, (i + 1) * DH)
            pr = T.functional.softmax((q_all[:, sl] @ k_all[:, sl].T) * inv + mask, axis=-1)
            part = (pr @ v_all[:, sl]) @ Wo[:, sl].T
            acc = part if acc is None else acc + part
        return acc

    x = T.tensor(x0)
    counts.clear(); a = per_head(x); rt.sync(); c_head = sum(counts.values())
    counts.clear(); b = fused(x); rt.sync(); c_fused = sum(counts.values())
    va = np.asarray(a.data, np.float32); vb = np.asarray(b.data, np.float32)
    rel = float(np.abs(va - vb).max() / max(1e-9, float(np.abs(va).max())))
    log("块输出相对差 = %.2e" % rel)
    log("发射次数：per-head %d，全宽 %d（少 %d，%.1f%%）"
        % (c_head, c_fused, c_head - c_fused, 100.0 * (c_head - c_fused) / max(c_head, 1)))

    def timeit(fn):
        fn(x); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(REP):
            fn(x)
        torch.cuda.synchronize()
        return (time.time() - t0) / REP

    t_head, t_fused = timeit(per_head), timeit(fused)
    log("耗时：per-head %.2f ms，全宽 %.2f ms（省 %.2f ms，%.1fx）"
        % (t_head * 1000, t_fused * 1000, (t_head - t_fused) * 1000, t_head / max(t_fused, 1e-9)))
    log("结论：%s" % ("PASS（等价）" if rel <= 1e-5 else "FAIL（不等价，不能改）"))
    st.close()
    rt.empty_cache()


if __name__ == "__main__":
    main()
