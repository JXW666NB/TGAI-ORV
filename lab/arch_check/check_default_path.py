# -*- coding: utf-8 -*-
"""默认路径逐位不变：旧版（git HEAD）vs 新版 `v3_train_cli`。

CPU 上跑（不需要 GPU）。比三样东西：

1. **参数指纹**：`params_of()` 顺序下每个张量的 (名字, 形状, float32 字节的 sha256)，
   逐项**全等**（不是"差不多"）。这是任务里说的"像 csa-integ 那样验证 323 个参数全等"。
2. **前向 logits**：同一份权重、同一批 ids，`logits` 的 float32 字节 sha256 全等。
3. **新开关的增量**：`--tie-embeddings` 少 1 个参数、`--qk-norm` 每 Global 层多 2 个、
   `--moe-post-norm` 每层多 1 个、`--pad-id` 把该行压成 0、`--attn-res` 每层多 4 个。

用法（仓库根）：

    python -X utf8 _arch_check/check_default_path.py
"""
import hashlib
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

import tgai_go.train as T                                        # noqa: E402
from tgai_go.train import v3_train_cli as NEW                    # noqa: E402

T.set_default_device("cpu")


def load_old():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "old_v3_train_cli.py")
    spec = importlib.util.spec_from_file_location("old_v3_cli", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def patch_cpu_moe():
    """把 `v3_moe.moe_dispatch_orv` 换成 numpy 版（**只为 CPU 上前向对拍**）。

    `v3_moe` 的散回走运行时内核 `scatter_add_rows`（CUDA only），CPU 上没有。
    `v3_moe` 自带同数学的 numpy 参考 `moe_dispatch_numpy`，这里用它顶替 ——
    两侧（旧版/新版）走同一个被顶替的函数，所以"逐位相等"比较的仍然是**代码路径**的
    差异，不是数学差异。注意：这只影响本脚本，不改任何包内文件。
    """
    import tgai_go.train.v3_moe as _M

    def dispatch(x, eids_np, gate, W1, W2, W3, beta1=4.0, beta2=25.0):
        xn = np.asarray(x.to_numpy(), np.float32)
        gn = np.asarray(gate.to_numpy(), np.float32).reshape(-1)
        Ws = [[np.asarray(w.to_numpy(), np.float32) for w in W]
              for W in (W1, W2, W3)]
        out = _M.moe_dispatch_numpy(xn, eids_np, gn, Ws[0], Ws[1], Ws[2],
                                    beta1, beta2)
        return T.tensor(out)

    _M.moe_dispatch_orv = dispatch
    return dispatch


def orv_named(P):
    """(name, tensor)；名字口径与 `params_of` 的摊平顺序一致（用于报告）。"""
    yield "emb", P["emb"]
    yield "ln_f", P["ln_f"]
    yield "head", P["head"]
    for i, L in enumerate(P["layers"]):
        for k, v in L.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    yield "layers.%d.csa.%s" % (i, kk), vv
            elif isinstance(v, list):
                for j, vv in enumerate(v):
                    yield "layers.%d.%s.%d" % (i, k, j), vv
            else:
                yield "layers.%d.%s" % (i, k), v


def fingerprint(P):
    rows, seen = [], set()
    for n, t in orv_named(P):
        if t is None:
            rows.append((n, None, None, 0))
            continue
        if id(t) in seen:                      # tie：同一个张量只指纹一次
            rows.append((n, "tied", tuple(int(s) for s in t.shape), None))
            continue
        seen.add(id(t))
        arr = np.ascontiguousarray(np.asarray(t.to_numpy(), np.float32))
        rows.append((n, tuple(int(s) for s in t.shape),
                     hashlib.sha256(arr.tobytes()).hexdigest()[:16],
                     int(np.asarray(T.params_of(P) if False else arr).size)))
    return rows


def flat_params(P):
    return [np.ascontiguousarray(np.asarray(t.to_numpy(), np.float32))
            for t in NEW.params_of(P)]


CFG = dict(d=48, d_ff=64, E=4, layers=4, n_q=4, n_kv=2, vocab=128, seq=16,
           steps=1, attn_res=False, attn_res_block=2)


def old_forward(old, P, cfg, cos, sin, ids, top_k=1):
    """旧版 main() 的层循环（默认路径的逐字复刻，只用于对拍）。"""
    V = old.V
    MOE = old.MOE
    seq, d = len(ids), cfg["d"]
    n_kv_attn = int(cfg.get("n_kv_attn", cfg["n_kv"]))
    x = P["emb"][ids].reshape(1, seq, d)
    for L in P["layers"]:
        xn = V.rmsnorm_orv(x, L["ln1"])
        attn = V.attention_orv(xn, L["wq"], L["wk"], L["wv"], L["wo"], cos, sin,
                               d // cfg["n_q"], cfg["n_q"], n_kv_attn,
                               lowmem=False, flash=False)
        h = x + attn
        hn = V.rmsnorm_orv(h, L["ln2"]).reshape(seq, d)
        if top_k == 1:
            eids, g, _s = MOE.route_top1(hn, L["router"])
            mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
        else:
            eids, gates, _s = MOE.route_topk(hn, L["router"], top_k)
            mo = MOE.moe_topk_orv(hn, eids, gates, L["W1"], L["W2"], L["W3"])
        mo = mo + V.situ_glu_orv(hn, L["S1"], L["S2"], L["S3"])
        x = h + mo.reshape(1, seq, d)
    xf = V.rmsnorm_orv(x, P["ln_f"])
    return xf.reshape(seq, d) @ P["head"].transpose(0, 1)


def main():
    old = load_old()
    patch_cpu_moe()
    fails = []

    # ── 1. 默认路径：参数指纹 ──
    rng_a = np.random.default_rng(0)
    P_old = old.build(dict(CFG), rng_a)
    rng_b = np.random.default_rng(0)
    P_new = NEW.build(dict(CFG), rng_b)
    fp_old, fp_new = fingerprint(P_old), fingerprint(P_new)
    if [r[:3] for r in fp_old] != [r[:3] for r in fp_new]:
        fails.append("参数指纹不一致（名字/形状/hash）")
        for a, b in zip(fp_old, fp_new):
            if a[:3] != b[:3]:
                print("  %-28s old=%s new=%s" % (a[0], a[1:3], b[1:3]))
    n_tensors = len([r for r in fp_new if r[1] not in (None, "tied")])
    print("[1] 参数指纹：%d 个名字 / %d 个张量，旧版→新版 %s"
          % (len(fp_new), n_tensors, "全等" if not fails else "不一致"))
    hs = hashlib.sha256("".join("%s|%s|%s" % (r[0], r[1], r[2]) for r in fp_new)
                        .encode()).hexdigest()
    print("    汇总指纹 sha256=%s" % hs[:32])

    # ── 2. 默认路径：前向 logits 逐位相等 ──
    cos_np, sin_np = NEW.V.rope_cos_sin(CFG["seq"], CFG["d"] // CFG["n_q"], 10000.0)
    cos, sin = T.tensor(cos_np), T.tensor(sin_np)
    rng_ids = np.random.default_rng(7)
    ids = rng_ids.integers(0, CFG["vocab"], CFG["seq"]).astype(np.int64)
    lg_old = old_forward(old, P_old, CFG, cos, sin, ids, top_k=1)

    class _O(object):
        top_k = 1
        shared = True
        chunk = 0
        lowmem = False
        flash = False

    lg_new = NEW.forward_logits(P_new, dict(CFG), NEW.layer_kinds(CFG), None,
                                cos, sin, ids, opts=_O())
    b_old = np.ascontiguousarray(np.asarray(lg_old.to_numpy(), np.float32)).tobytes()
    b_new = np.ascontiguousarray(np.asarray(lg_new.to_numpy(), np.float32)).tobytes()
    same = hashlib.sha256(b_old).hexdigest() == hashlib.sha256(b_new).hexdigest()
    max_abs = float(np.abs(np.asarray(lg_old.to_numpy(), np.float64)
                           - np.asarray(lg_new.to_numpy(), np.float64)).max())
    print("[2] 默认路径 logits：旧版 sha=%s 新版 sha=%s → %s（max_abs=%.3e）"
          % (hashlib.sha256(b_old).hexdigest()[:16],
             hashlib.sha256(b_new).hexdigest()[:16],
             "逐位相等" if same else "不同", max_abs))
    if not same:
        fails.append("默认路径前向不逐位相等（max_abs=%.3e）" % max_abs)

    # ── 3. 新开关的增量（数量与形状；不要求与旧版相等，要求"确实变了且变对"） ──
    base_n = len(NEW.params_of(P_new))
    cases = {}

    c = dict(CFG, tie=True)
    p = NEW.build(c, np.random.default_rng(0))
    ok = p["head"] is p["emb"]
    cases["tie"] = (len(NEW.params_of(p)), base_n - 1, ok)

    c = dict(CFG, qk_norm=True)
    p = NEW.build(c, np.random.default_rng(0))
    has = all(("qn" in L and "kn" in L) for L in p["layers"])
    cases["qk_norm"] = (len(NEW.params_of(p)), base_n + 2 * CFG["layers"], has)

    c = dict(CFG, moe_post_norm=True)
    p = NEW.build(c, np.random.default_rng(0))
    has = all("post_norm" in L for L in p["layers"])
    cases["moe_post_norm"] = (len(NEW.params_of(p)), base_n + CFG["layers"], has)

    c = dict(CFG, pad_id=3)
    p = NEW.build(c, np.random.default_rng(0))
    row = np.asarray(p["emb"].to_numpy())[3]
    cases["pad_id"] = (len(NEW.params_of(p)), base_n, float(np.abs(row).max()) == 0.0)

    c = dict(CFG, attn_res=True, attn_res_block=2)
    p = NEW.build(c, np.random.default_rng(0))
    has = all(all(k in L for k in ("res_w_attn", "res_gn_attn",
                                   "res_w_mlp", "res_gn_mlp")) for L in p["layers"])
    cases["attn_res"] = (len(NEW.params_of(p)), base_n + 4 * CFG["layers"], has)

    for name, (got, want, flag) in cases.items():
        good = (got == want) and flag
        print("[3] %-14s 参数 %d（期望 %d，%s），语义标志 %s → %s"
              % (name, got, want, "=" if got == want else "≠",
                 flag, "OK" if good else "FAIL"))
        if not good:
            fails.append("%s 的参数增量/语义不符" % name)

    print("\n==> %s" % ("ALL OK" if not fails else "FAILURES: %r" % (fails,)))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
