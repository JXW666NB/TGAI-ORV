# -*- coding: utf-8 -*-
"""锁定异常模块：真模型上逐张量比较 amp 开/关的梯度范数。

用 v3_train_cli 的公开 API：build / params_of / forward_logits / layer_kinds。
两次运行用同一 rng 种子 ⇒ build 顺序一致 ⇒ 可按**位置**对应参数。

用法（V100）：
    D:\\orv_pip_0962\\Scripts\\python.exe model_grad_diff.py
"""
import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_ops as V
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()

CFG = dict(d=1024, d_ff=2816, E=2, layers=2, n_q=16, n_kv=2, vocab=96000, seq=512,
           attn_res=False, attn_res_block=2, top_k=2, csa=4, n_kv_attn=16,
           rope_theta=500000.0, csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
SEQ = CFG["seq"]
DH = CFG["d"] // CFG["n_q"]


def loss_and_grads(amp_on):
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
    rng = np.random.default_rng(1234)
    P = CLI.build(CFG, rng)
    kinds = CLI.layer_kinds(CFG)
    params = CLI.params_of(P)
    plan = None
    if int(CFG.get("csa", 0) or 0) > 0:
        from tgai_go.train import v3_csa as C
        csac = CFG["csa_cfg"]
        plan = C.make_plan(csac, 1, SEQ)
        cos, sin = C.rope_tables(csac, SEQ)
    else:
        cos, sin = V.rope_cos_sin(SEQ, DH, float(CFG["rope_theta"]))
        cos, sin = T.tensor(cos), T.tensor(sin)
    ids = np.random.default_rng(7).integers(0, CFG["vocab"], SEQ).astype(np.int64)
    tgt = np.roll(ids, -1)

    if amp_on:
        T.amp_on(dtype="fp16", weight_cache=True)
    logits = CLI.forward_logits(P, CFG, kinds, plan, cos, sin, ids)
    loss = T.functional.cross_entropy(logits.reshape(SEQ, CFG["vocab"]), tgt)
    for p in params:
        p.grad = None
    loss.backward()
    rt.sync()
    lv = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
    gn = []
    for p in params:
        g = getattr(p, "grad", None)
        gn.append(0.0 if g is None
                  else float(np.linalg.norm(np.asarray(g.to_numpy(), np.float64))))
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
    return lv, gn, [tuple(getattr(p, "shape", ())) for p in params]


def main():
    lv_off, g_off, shapes = loss_and_grads(False)
    lv_on, g_on, _ = loss_and_grads(True)
    print("loss  off=%.6f  amp=%.6f  (差 %.3e)" % (lv_off, lv_on, abs(lv_off - lv_on)))
    rows = [(b / a, i, a, b) for i, (a, b) in enumerate(zip(g_off, g_on)) if a > 0]
    rows.sort(reverse=True)
    print("\n=== 按 amp/off 梯度范数比值排序（前 12）===")
    for r, i, a, b in rows[:12]:
        print("idx=%-3d shape=%-18s ratio=%10.3f  off=%9.3e  amp=%9.3e"
              % (i, str(shapes[i]), r, a, b))
    print("\n全局: off=%.6e  amp=%.6e  ratio=%.3f"
          % (np.sqrt(sum(v * v for v in g_off)), np.sqrt(sum(v * v for v in g_on)),
             (np.sqrt(sum(v * v for v in g_on)) /
              max(np.sqrt(sum(v * v for v in g_off)), 1e-30))))


if __name__ == "__main__":
    main()
