# -*- coding: utf-8 -*-
"""H1 判别实验：`_sgemm_f16` 的临时量生命周期。

假设 H1：`_sgemm_f16` 内部 `self.empty()` 出的 a16/b16 在 GEMM 入队后立刻掉引用、
归还池子；cuBLAS 是异步的 ⇒ 池子把同一块内存再分给别人 → 数据被覆盖。

判别：给 `_sgemm_f16` 打补丁，**每次调用后强制同步**（等 GEMM 落地才还引用）。
      若梯度爆炸消失 ⇒ H1 命中。
"""
import numpy as np

import tgai_go.train as T
from tgai_go.train import cuda as cudamod
from tgai_go.train import v3_ops as V
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()

# 全尺寸配置（scale_bisect 里复现 4130 倍的那个）
CFG = dict(d=1024, d_ff=2816, E=2, layers=2, n_q=16, n_kv=2, vocab=96000, seq=512,
           attn_res=False, attn_res_block=2, top_k=2, csa=4, n_kv_attn=16,
           rope_theta=500000.0, csa_compress=4, csa_conv_kernel=3, csa_local_window=0)

CLS = type(rt)      # _CudaAPI 在 cuda 模块里是**实例**，类要从实例上取


def _find_orig():
    for name in ("_sgemm_f16", "sgemm_f16"):
        if hasattr(CLS, name):
            return name, getattr(CLS, name)
    return None, None


NAME, ORIG = _find_orig()
if ORIG is None:
    raise SystemExit("找不到 _sgemm_f16，无法做 H1 判别；修法：确认 cuda.py 里 fp16 分支的函数名")


def _patched_sync(self, *a, **kw):
    r = ORIG(self, *a, **kw)
    self.sync()          # ★ 等 GEMM 真正完成，再让临时量被回收
    return r


def run(patch, label):
    if patch:
        setattr(CLS, NAME, _patched_sync)
    else:
        setattr(CLS, NAME, ORIG)
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
    rng = np.random.default_rng(1234)
    cfg = dict(CFG)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    seq = int(cfg["seq"])
    from tgai_go.train import v3_csa as C
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
    lv = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
    tot = 0.0
    for p in params:
        g = getattr(p, "grad", None)
        if g is not None:
            a = np.asarray(g.to_numpy(), np.float64)
            tot += float((a * a).sum())
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
    print("%-22s loss=%.6f  gnorm=%.6e" % (label, lv, np.sqrt(tot)))
    return lv, float(np.sqrt(tot))


def main():
    print("=== H1 判别（全尺寸配置，csa=4 / d=1024 / seq=512）===")
    print("fp16 分支函数:", NAME)
    l_fp32, g_fp32 = run(False, "amp_off（fp32 基线）")
    # amp 但用未打补丁的 _sgemm_f16：需要先跑一次 fp32 才能对比 —— 这里直接开 amp
    T.amp_on(dtype="fp16", weight_cache=True)
    l_a, g_a = run(False, "amp_on（未打补丁）")
    l_b, g_b = run(True, "amp_on + 每次同步")
    print()
    print("未打补丁 / fp32 基线 = %.1f 倍" % (g_a / max(g_fp32, 1e-30)))
    print("打补丁后 / 未打补丁  = %.4f" % (g_b / max(g_a, 1e-30)))
    print("判定：%s" % ("H1 命中（同步后恢复正常）" if abs(g_b / max(g_a, 1e-30) - 1.0) < 0.35
                     else "H1 未命中（同步没解决问题）"))


if __name__ == "__main__":
    main()
