# -*- coding: utf-8 -*-
"""噪声底测量（双进程版）：单条训练路径，把 loss 曲线写 json。

用法：
    python nf_path.py --path graph     --out D:\\tgai_orv\\nf_graph.json
    python nf_path.py --path layerwise --out D:\\tgai_orv\\nf_layer.json

为什么分两个进程：ORV 有 bf16 影子缓存，`load_all` 复位在本机实测不可靠
（同进程先后跑两条路径时，第二条第 0 步 loss 就是"训练后"的水平）。
两个进程各自冷启动从 `.tg` 载入 ⇒ 初始权重天然完全一致。
"""
import argparse
import json
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=["graph", "layerwise"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import glob
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
    opts = [Fp16MomAdamW(tr.params_of(i), **kw) for i in range(NL)]
    head_opt = Fp16MomAdamW(tr.head_params(), **kw)

    losses = []
    for step in range(STEPS):
        if args.path == "graph":
            x = T.tensor(x0, requires_grad=True)
            for i in range(NL):
                x = tr.fwd(x, i, cos, sin)
            lg = tr.head(tr.final_norm(x))
            ls = T.functional.cross_entropy(lg, ids[1:])
            losses.append(float(np.asarray(ls.to_numpy() if hasattr(ls, "to_numpy") else ls)))
            ls.backward()
            for o in opts:
                o.step(); o.zero_grad()
            head_opt.step(); head_opt.zero_grad()
            for p in tr.head_params():
                p.grad = None
        else:
            xs = tr.forward_all(x0, cos, sin)
            ls, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
            losses.append(float(np.asarray(ls.to_numpy() if hasattr(ls, "to_numpy") else ls)))
            tr.backward_and_step(xs, g, cos, sin, opts)
            head_opt.step(); head_opt.zero_grad()
            for p in tr.head_params():
                p.grad = None

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"path": args.path, "losses": losses, "nl": NL, "seq": TL,
                   "lr": LR, "steps": STEPS}, f, ensure_ascii=False, indent=1)
    log("%s 完成：首步 %.6f 末步 %.6f → %s" % (args.path, losses[0], losses[-1], args.out))
    st.close()


if __name__ == "__main__":
    main()
