# -*- coding: utf-8 -*-
"""确认 `_sgemm` 真正归属的模块，并在**正确的模块字典**上改门控。

上一轮 `CM._FP16_MIN_OUTPUT = 1e9` 完全无效（fp16 分支仍 79 次）⇒
很可能 `cuda` 是包的转发壳，`_sgemm` 的真实 `__globals__` 在别的模块。
"""
import sys

import numpy as np

import tgai_go.train as T
from tgai_go.train import cuda as CM
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)
MOD = sys.modules[CLS.__module__]

print("cuda 模块文件      :", getattr(CM, "__file__", "?"))
print("rt 的类模块名      :", CLS.__module__)
print("该模块文件         :", getattr(MOD, "__file__", "?"))
print("是同一个模块吗     :", MOD is CM)
print("MOD 里有门控常量吗 :", hasattr(MOD, "_FP16_MIN_OUTPUT"))

CFG = dict(d=1024, d_ff=2816, E=2, layers=2, n_q=16, n_kv=2, vocab=96000, seq=512,
           attn_res=False, attn_res_block=2, top_k=2, csa=4, n_kv_attn=16,
           rope_theta=500000.0, csa_compress=4, csa_conv_kernel=3, csa_local_window=0)

CNT = {"f16": 0}
_orig = getattr(CLS, "_sgemm_f16")


def _wrapped(self, *a, **kw):
    CNT["f16"] += 1
    return _orig(self, *a, **kw)


setattr(CLS, "_sgemm_f16", _wrapped)


def run(min_out, label):
    CNT["f16"] = 0
    for m in (CM, MOD):
        if hasattr(m, "_FP16_MIN_OUTPUT"):
            m._FP16_MIN_OUTPUT = int(min_out)
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
    print("%-26s gnorm=%.6e  fp16 调用=%d" % (label, float(np.sqrt(tot)), CNT["f16"]))
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()


def main():
    run(262144, "MIN=262144（默认）")
    run(10 ** 9, "MIN=1e9（应 0 次）")
    run(1, "MIN=1（应更多）")


if __name__ == "__main__":
    main()
