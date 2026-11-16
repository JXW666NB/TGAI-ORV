# -*- coding: utf-8 -*-
"""数一数：amp 打开后，`_sgemm` 的三个低精度分支各被走了多少次。

A. fp32 基线（amp 关）
B. amp 开

若 B 里 `sgemm_split16` / `_sgemm_bf16` 的计数明显非零，
说明真正生效的不是 fp16 分支 —— 这就能解释"关掉 fp16 门控也没用"。
"""
import numpy as np

import tgai_go.train as T
from tgai_go.train import cuda as CM
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)

CFG = dict(d=1024, d_ff=2816, E=2, layers=2, n_q=16, n_kv=2, vocab=96000, seq=512,
           attn_res=False, attn_res_block=2, top_k=2, csa=4, n_kv_attn=16,
           rope_theta=500000.0, csa_compress=4, csa_conv_kernel=3, csa_local_window=0)

COUNTS = {}


def _wrap(name):
    orig = getattr(CLS, name, None)
    if orig is None:
        return None

    def fn(self, *a, **kw):
        COUNTS[name] = COUNTS.get(name, 0) + 1
        return orig(self, *a, **kw)
    setattr(CLS, name, fn)
    return orig


def run(amp, label, min_out=262144):
    CM._FP16_MIN_OUTPUT = int(min_out)
    COUNTS.clear()
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
    rng = np.random.default_rng(1234)
    cfg = dict(CFG)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    seq = int(cfg["seq"])
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(7).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)
    if amp:
        T.amp_on(dtype="fp16", weight_cache=True)
    logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
    loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
    for p in params:
        p.grad = None
    loss.backward()
    rt.sync()
    tot = 0.0
    for p in params:
        g = getattr(p, "grad", None)
        if g is not None:
            a = np.asarray(g.to_numpy(), np.float64)
            tot += float((a * a).sum())
    st = T.amp_state()
    print("%-14s gnorm=%.6e  分支调用次数: %s" % (label, float(np.sqrt(tot)),
                                              dict(sorted(COUNTS.items()))))
    print("               amp_state: on=%s fp16=%s bf16=%s c32=%s"
          % (st.get("on"), st.get("fp16"), st.get("bf16"), st.get("c32")))
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()


def main():
    names = ["_sgemm_f16", "_sgemm_bf16", "sgemm_split16", "_sgemm", "gemm_f16_nt_raw",
             "gemm_f16_nn_raw", "gemm_f16_tn_raw"]
    wrapped = [n for n in names if _wrap(n)]
    print("已插桩:", wrapped)
    print()
    run(False, "A amp_off")
    run(True, "B amp_on 默认")
    run(True, "C amp_on MIN=1e9", min_out=10 ** 9)
    run(True, "D amp_on MIN=1", min_out=1)


if __name__ == "__main__":
    main()
