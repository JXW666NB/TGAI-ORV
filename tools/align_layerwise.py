# -*- coding: utf-8 -*-
"""对齐定位 ⑩：**ORV 内部自比** —— 逐层重算反向 vs 图反向，梯度是否等价？

已排除：前向（逐位相同）、优化器（≤5e-8）、路由、分组 kernel。
剩下唯一未验证的是训练用的 **逐层重算反向**（`backward_and_step`）与图反向是否给出同样梯度。
本脚本用 ORV 自己的算子把两条路都跑一遍（不引入第三方实现）：

  路 A（图反向）：整层建图 → (y·g).sum().backward() → 读参数梯度
  路 B（逐层重算）：用 forward_all 缓存的 xs[i] 重新建一次图 → 同样 backward → 读梯度
  两者数学上应当相同；若不同，说明"重算"引入了差异（例如 xs[i] 与图上输入不同、或缓存被改）。

运行（需独占 GPU）：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_layerwise.py
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
SEQ = 64


def rel(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def main():
    import glob
    import json
    import os
    import tgai_go.train as T
    from tgai_go.bpe import BOS_ID, UNK_ID, BpeTokenizer
    from tgai_go.read_tg import close_tg, read_tg
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
    tr = V27Trainer(cfg, n_layers=1)
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
    L = tr.layers[0]

    # 先拿到上游梯度 g（来自 head 反向，路径与训练一致）
    xs = tr.forward_all(x0, cos, sin)
    loss, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
    g_np = np.asarray(g, np.float32)
    log("head 反向得到上游梯度 g，|g|max=%.4f" % float(np.abs(g_np).max()))

    def grab():
        out = {}
        for nm, po in (("ln1", L["ln1"].weight), ("ln2", L["ln2"].weight),
                       ("q0", L["q"][0].weight), ("o0", L["o"][0].weight),
                       ("router", L["moe"].router.weight),
                       ("e0.gate", L["moe"].experts[0].gate.weight),
                       ("e0.down", L["moe"].experts[0].down.weight)):
            gr = getattr(po, "grad", None)
            out[nm] = None if gr is None else np.asarray(
                gr.to_numpy() if hasattr(gr, "to_numpy") else gr, np.float32)
        return out

    def clear():
        for p in tr.params_of(0):
            p.grad = None

    # ---- 路 A：直接在 xs[0] 上重建（一次）----
    clear()
    x_in = T.tensor(np.ascontiguousarray(xs[0]), requires_grad=True)
    y = tr.fwd(x_in, 0, cos, sin)
    (y * T.tensor(g_np)).sum().backward()
    gA = grab()
    gA_up = np.asarray(x_in.grad.to_numpy(), np.float32) if x_in.grad is not None else None

    # ---- 路 B：用 forward_all 缓存的同一 xs[0]（模拟逐层重算）----
    clear()
    xs2 = tr.forward_all(x0, cos, sin)
    x_in2 = T.tensor(np.ascontiguousarray(xs2[0]), requires_grad=True)
    y2 = tr.fwd(x_in2, 0, cos, sin)
    (y2 * T.tensor(g_np)).sum().backward()
    gB = grab()
    gB_up = np.asarray(x_in2.grad.to_numpy(), np.float32) if x_in2.grad is not None else None

    log("=== ORV 内部：路A(直接重建) vs 路B(forward_all 缓存) ===")
    log("  输入 xs[0] 两路是否相同: max|Δ|=%.3e"
        % float(np.abs(np.asarray(xs[0], np.float32) - np.asarray(xs2[0], np.float32)).max()))
    worst = (0.0, "")
    for nm in ("ln1", "ln2", "q0", "o0", "router", "e0.gate", "e0.down"):
        if gA[nm] is None or gB[nm] is None:
            log("  %-9s 缺梯度" % nm)
            continue
        r = rel(gA[nm], gB[nm])
        if r > worst[0]:
            worst = (r, nm)
        log("  %-9s 相对=%.2e  %s" % (nm, r, "一致" if r < 1e-6 else "★分叉"))
    if gA_up is not None and gB_up is not None:
        log("  上游 d(x0) 相对=%.2e" % rel(gA_up, gB_up))
    log("最差：%s 相对=%.2e" % (worst[1], worst[0]))

    # ---- 附加：与图反向（整链）比一次，看逐层近似是否引入偏差 ----
    clear()
    x = T.tensor(x0, requires_grad=True)
    yy = tr.fwd(x, 0, cos, sin)
    xt = T.tensor(np.ascontiguousarray(yy.to_numpy()), requires_grad=True)
    lg = tr.head(tr.final_norm(xt))
    ls = T.functional.cross_entropy(lg, ids[1:])
    ls.backward()   # head 的梯度
    (yy * T.tensor(g_np)).sum().backward()   # 用同一 g 反传到层
    gC = grab()
    log("=== 附加：整链图反向 vs 逐层 ===")
    for nm in ("ln1", "ln2", "q0", "o0", "router", "e0.gate", "e0.down"):
        if gC[nm] is None or gA[nm] is None:
            continue
        log("  %-9s 相对=%.2e" % (nm, rel(gC[nm], gA[nm])))
    st.close()


if __name__ == "__main__":
    main()
