# -*- coding: utf-8 -*-
"""对齐定位 ⑪：**噪声底**（同侧自比）—— ORV 图反向 vs ORV 逐层反向，各跑 15 步。

为什么必须做：前向逐位相同、梯度 ≤7e-6、AdamW ≤5e-8、逐层反算与图反向 ≤3.7e-7，
但 ORV 与 torch 的 15 步曲线仍差 ~3e-03。
若**同侧两条数学等价路径**之间的差也有这个量级，那 3e-03 就是本设置下的**噪声底**
（loss 曲面很陡，浮点求和顺序差异被放大），判据应写成"在噪声底内"，而不是硬卡 1e-4。

关键细节：重置权重必须同时调 `fp16_shadows_invalidate()`，
否则影子缓存是旧的 —— 上一轮 P2 出现 32% 假象就是这个原因。

运行（需独占 GPU）：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_noise_floor.py
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
SEQ, NL, STEPS, LR = 128, 4, 15, 1e-5
BETAS, EPS, WD = (0.9, 0.999), 1e-8, 0.0


def main():
    import glob
    import json
    import os
    import tgai_go.train as T
    from tgai_go.bpe import BOS_ID, UNK_ID, BpeTokenizer
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW
    from tgai_go.train.v27_train import TGState, V27Trainer

    info = read_tg(MODEL_TG)
    h = info["header"]
    tk = BpeTokenizer(info["tokens"], info["merges"], unk_id=UNK_ID, bos_id=BOS_ID)
    close_tg(info)
    DM, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = DM // NH, float(h["rope_theta"])
    cfg = dict(d_model=DM, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    tr.attach_head(st)
    emb = st["token_embedding.weight"]

    ids = None
    for p in sorted(glob.glob(os.path.join("D:" + chr(92) + "SFT" + chr(0x8bed) + chr(0x6599) + chr(0x5e93),
                                          "**", "*.jsonl"), recursive=True)):
        with open(p, encoding="utf-8", errors="replace") as f:
            for _ in range(40):
                line = f.readline()
                if not line:
                    break
                try:
                    t = (json.loads(line).get("text") or "")
                except Exception:
                    continue
                if len(t) > 600:
                    ids = np.asarray(tk.encode(t, add_bos=True)[:SEQ + 1], np.int64)
                    break
        if ids is not None:
            break
    TL = len(ids) - 1
    x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(TL, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32)); sin = T.tensor(np.sin(ang).astype(np.float32))
    kw = dict(lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, fp32_state=True)

    def reset():
        tr.load_all(st)
        tr.attach_head(st)
        try:
            T.fp16_shadows_invalidate()
        except Exception as e:
            log("!! 影子失效失败：%s" % type(e).__name__)
        try:
            T._bf16_step_boundary()
        except Exception:
            pass
        T._cuda().empty_cache()

    # ---------- A：图反向 ----------
    opts = [Fp16MomAdamW(tr.params_of(i), **kw) for i in range(NL)]
    head = Fp16MomAdamW(tr.head_params(), **kw)
    lossA = []
    for step in range(STEPS):
        x = T.tensor(x0, requires_grad=True)
        for i in range(NL):
            x = tr.fwd(x, i, cos, sin)
        lg = tr.head(tr.final_norm(x))
        ls = T.functional.cross_entropy(lg, ids[1:])
        lossA.append(float(np.asarray(ls.to_numpy() if hasattr(ls, "to_numpy") else ls)))
        ls.backward()
        for o in opts:
            o.step(); o.zero_grad()
        head.step(); head.zero_grad()
        for p in tr.head_params():
            p.grad = None

    reset()

    # ---------- B：逐层重算反向 ----------
    opts2 = [Fp16MomAdamW(tr.params_of(i), **kw) for i in range(NL)]
    head2 = Fp16MomAdamW(tr.head_params(), **kw)
    lossB = []
    for step in range(STEPS):
        xs = tr.forward_all(x0, cos, sin)
        lsm, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
        lossB.append(float(np.asarray(lsm.to_numpy() if hasattr(lsm, "to_numpy") else lsm)))
        tr.backward_and_step(xs, g, cos, sin, opts2)
        head2.step(); head2.zero_grad()
        for p in tr.head_params():
            p.grad = None

    print("\n%5s %14s %14s %12s %10s" % ("step", "ORV 图反向", "ORV 逐层", "|Δ|", "相对"))
    worst = 0.0
    for i, (a, b) in enumerate(zip(lossA, lossB)):
        d = abs(a - b)
        r = d / max(1e-9, abs(b))
        worst = max(worst, r)
        print("%5d %14.6f %14.6f %12.2e %9.2e" % (i, a, b, d, r))
    print("\n同侧自比最差相对偏差 = %.2e（这就是本设置下的噪声底）" % worst)
    st.close()


if __name__ == "__main__":
    main()
