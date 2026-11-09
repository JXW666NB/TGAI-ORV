# -*- coding: utf-8 -*-
"""fused vs 默认 attention 块：前向一致性 + 发射数 + 耗时 + 梯度一致性（串行、单 trainer）。

需先应用补丁：python tools/apply_fused_heads.py
运行（需独占 GPU）：python fused_block_full.py
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

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"
SEQ = 64
REP = 5


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

    counts = Counter()
    orig = type(rt).launch

    def counted(self, name, *a, **kw):
        counts[name] += 1
        return orig(self, name, *a, **kw)

    type(rt).launch = counted

    x0 = np.random.default_rng(0).normal(0, 1, (SEQ, D)).astype(np.float32)
    gw = np.random.default_rng(1).normal(0, 1, (SEQ, D)).astype(np.float32)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos_np, sin_np = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    def block(tr, L, back):
        cos = T.tensor(cos_np)
        sin = T.tensor(sin_np)
        x = T.tensor(x0)
        hh = L["ln1"](x)
        inv = 1.0 / np.sqrt(DH)
        mask = tr._causal_mask(SEQ)
        acc = None
        if getattr(tr, "fused_heads", False):
            cos_t = T.tensor(np.tile(cos_np, (1, NH)))
            sin_t = T.tensor(np.tile(sin_np, (1, NH)))
            q_all = T._rope_composed(L["q"][0](hh), cos_t, sin_t)
            k_all = T._rope_composed(L["k"][0](hh), cos_t, sin_t)
            v_all = L["v"][0](hh)
            for i in range(NH):
                sl = slice(i * DH, (i + 1) * DH)
                pr = T.functional.softmax((q_all[:, sl] @ k_all[:, sl].T) * inv + mask, axis=-1)
                part = (pr @ v_all[:, sl]) @ L["o"][0].weight[:, sl].T
                acc = part if acc is None else acc + part
        else:
            for i in range(NH):
                q = T._rope_composed(L["q"][i](hh), cos, sin)
                k = T._rope_composed(L["k"][i](hh), cos, sin)
                v = L["v"][i](hh)
                pr = T.functional.softmax((q @ k.T) * inv + mask, axis=-1)
                part = L["o"][i](pr @ v)
                acc = part if acc is None else acc + part
        if back:
            (acc * T.tensor(gw)).sum().backward()
        return acc

    def collect_grads(tr, L):
        """把 per-head 梯度拼成全宽，便于与 fused 比较。"""
        out = {}
        for key, axis in (("q", 0), ("k", 0), ("v", 0), ("o", 1)):
            mods = L[key]
            gs = []
            for m in mods:
                g = m.weight.grad
                if g is None:
                    return None
                gs.append(np.asarray(g.to_numpy(), np.float32))
            out[key] = gs[0] if len(gs) == 1 else np.concatenate(gs, axis=axis)
        return out

    def run(fused):
        st = TGState(MODEL_TG)
        tr = V27Trainer(cfg, n_layers=1)
        tr.fused_heads = bool(fused)
        tr.load_all(st)
        L = tr.layers[0]
        if fused and int(L["q"][0].weight.shape[0]) != D:
            log("!! 补丁未应用")
            st.close()
            return None
        block(tr, L, False); rt.sync()                   # 预热
        counts.clear()
        y = block(tr, L, True); rt.sync()
        launches = sum(counts.values())
        out = np.asarray(y.to_numpy(), np.float32).copy()
        grads = collect_grads(tr, L)
        # 计时（同一 trainer，串行，无额外分配）
        tray = []
        for _ in range(REP):
            rt.sync(); t0 = time.time()
            block(tr, L, False)
            rt.sync(); tray.append(time.time() - t0)
        st.close()
        rt.empty_cache()
        torch.cuda.synchronize()
        return out, launches, min(tray), grads

    a = run(False)
    if a is None:
        return 2
    b = run(True)
    if b is None:
        return 2
    (ya, la, ta, ga), (yb, lb, tb, gb) = a, b
    rel = float(np.abs(ya - yb).max() / max(1e-9, float(np.abs(ya).max())))
    log("前向：相对差 %.2e（判据 1e-6）" % rel)
    log("发射：默认 %d → fused %d（少 %d，%.1f%%）" % (la, lb, la - lb, 100.0 * (la - lb) / max(la, 1)))
    log("耗时：默认 %.2f ms → fused %.2f ms（省 %.2f ms/层 attention 块）"
        % (ta * 1000, tb * 1000, (ta - tb) * 1000))
    if ga is None or gb is None:
        log("梯度缺失，跳过比较")
        ok_g = False
    else:
        worst = 0.0
        for key in ("q", "k", "v", "o"):
            r = float(np.abs(ga[key] - gb[key]).max() / max(1e-9, float(np.abs(ga[key]).max())))
            worst = max(worst, r)
            log("  梯度 %s：相对差 %.2e（形状 %s）" % (key, r, ga[key].shape))
        ok_g = worst <= 1e-6
        log("梯度最差 %.2e（判据 1e-6）⇒ %s" % (worst, "PASS" if ok_g else "FAIL"))
    log("结论：%s" % ("PASS（前向+梯度）" if (rel <= 1e-6 and ok_g) else "FAIL"))
    return 0 if (rel <= 1e-6 and ok_g) else 1


if __name__ == "__main__":
    sys.exit(main())
