# -*- coding: utf-8 -*-
"""端到端数值对拍：ORV（tgai_go.train.v3_train_cli 同款前向）vs PyTorch（model_v3.TGAIV3Model）。

在一台同时有 torch 与 tgai-orv 0.9.84 的机器上跑（同一进程，ORV 由 sys.path 优先）：

    C:\\Python311\\python.exe D:\\orv_src\\tools\\e2e_parity.py --stage all

设计要点（为什么这样才对得上）
------------------------------
1. **权重只有一个来源**：先建 PyTorch 模型，把它的 `named_parameters()` 逐个映射进
   `v3_train_cli.build()` 造出的 ORV 参数字典。改完立刻断言「名字集合 + 形状」完全
   一致（多一个或少一个都报错），否则"同权重"比较不成立。
2. **配置逐项对齐**，不是只对齐形状。参考实现有些默认开关 ORV 没有对应参数（或语义
   不同），必须显式关掉/改成同一口径，否则"对不上"是配置差而非数学差。见 `ALIGNMENT`。
3. **loss 口径**：参考实现是 `logits[:, :-1]` vs `targets[:, 1:]`（丢掉最后一个位置）。
   CLI 的 `tgt = np.roll(ids, -1)` 会**多算一个环绕目标**（最后一个位置拿 ids[0] 当标签）。
   所以 ORV 侧主口径用 `logits[:T-1]` vs `ids[1:]`；CLI 原样口径单独打印一份对照。
4. **优化器**：两侧都显式传 lr/betas/eps/weight_decay，并把对象实际取值打进报告。
5. **自检**：`--stage selftest` 用同一 seed 复刻 CLI 的建参数/取数据顺序，把本脚本的
   前向结果与 `v3_train_cli.main()` 自己打印的 step-0 loss 对比 —— 证明"手工前向"
   与包里那条训练循环是同一个东西。

输出：人类可读块 + 末行 `=== JSON ===` 后跟完整 JSON。
"""
import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
import time

# ── 路径：ORV 0.9.84 优先，其次 V3 参考源码 ──
ORV_SP = os.environ.get("ORV_SP", r"D:\orv_pip_0962\Lib\site-packages")
V3_SRC = os.environ.get("TGAI_V3_SRC", r"D:\TGAI-V3-1.5B\03_src")
for _p in (V3_SRC, ORV_SP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

import model_v3 as M                                          # noqa: E402
import tgai_go.train as T                                     # noqa: E402
from tgai_go.train import v3_csa as CSA                       # noqa: E402
from tgai_go.train import v3_moe as MOE                       # noqa: E402
from tgai_go.train import v3_ops as V                         # noqa: E402
from tgai_go.train import v3_train_cli as CLI                 # noqa: E402

ALIGNMENT = [
    ("attn_qk_norm", "True (default)", "False",
     "ORV Global layers have no q/k RMSNorm params; CSA disabled via csa_cfg to match"),
    ("moe_balance", "'quantile' (default)", "'sign'",
     "ORV route_topk = softmax over all experts + raw affinity gate"),
    ("moe_post_norm", "True (default)", "False", "ORV MoE has no post-aggregation RMSNorm"),
    ("tie_embeddings", "True (default)", "False", "ORV build() keeps emb/head independent"),
    ("attn_res_enabled", "True (default)", "False", "ORV AttnRes lives in tools/ loop, not CLI forward"),
    ("pad_token_id", "256 (default)", "None", "padding_idx zeroes that row's grad; ORV has no such semantics"),
    ("dropout", "0.0", "0.0", "identical"),
    ("gradient_checkpointing", "off", "off", "reference default False; CLI loop has none"),
    ("ce_chunk", "0", "0", "same (non-chunked) CE code path"),
    ("dtype", "float32", "float32", "both fp32; ORV fp16_gemm stays False"),
]


# ═════════════════════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════════════════════
def orv_cfg(a, qk_norm=False, preset_csa_cfg=True):
    """与 `v3_train_cli.main()` 里 cfg 的构造逐字一致（只多一个 top_k 便于复用）。

    `preset_csa_cfg=False` 时**不**预置 `cfg["csa_cfg"]`，完全照 CLI 的行为（由 build()
    自己建，attn_qk_norm 取 CSAConfig 默认 True）—— selftest 必须走这条，否则
    init_weights 的 rng 抽样次数会变（多了 qn/kn 两次），连 ids 都会跟着漂。
    """
    cfg = dict(d=a.d, d_ff=a.d_ff, E=a.E, layers=a.layers,
               n_q=a.n_q, n_kv=2, vocab=a.vocab, seq=a.seq, steps=max(1, a.steps),
               attn_res=False, attn_res_block=2, top_k=a.top_k)
    if a.csa:
        cfg.update(csa=int(a.csa), n_kv_attn=a.n_q, rope_theta=500000.0,
                   csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
        if preset_csa_cfg:
            cfg["csa_cfg"] = CSA.CSAConfig(
                d_model=a.d, n_heads=a.n_q, n_kv_heads=0,
                csa_compress=cfg["csa_compress"], csa_latent=0,
                csa_conv_kernel=cfg["csa_conv_kernel"], csa_local_window=0,
                attn_qk_norm=bool(qk_norm))
    return cfg


def torch_cfg(a, qk_norm=False):
    return M.TGAIV3Config(
        vocab_size=a.vocab, d_model=a.d, n_layers=a.layers, n_heads=a.n_q,
        n_kv_heads=0, d_ff=a.d_ff, n_experts=a.E, n_activated=a.top_k,
        n_shared_experts=1, max_seq_len=max(a.seq, 64), dropout=0.0,
        pad_token_id=None, tie_embeddings=False, norm_eps=1e-6,
        attn_global_every=(int(a.csa) if a.csa else 1),
        csa_compress=4, csa_latent=0, csa_conv_kernel=3, csa_local_window=0,
        attn_qk_norm=bool(qk_norm), csa_seg_isolate=True, csa_causal_conv=True,
        attn_res_enabled=False, dense_attn_res_free=False,
        moe_balance='sign', moe_post_norm=False, moe_use_situ=True,
        moe_impl='mask', dense_ffn=False, rope_theta=500000.0, hetero_rope=False,
        qk_telemetry_every=1, ce_chunk=0,
    )


def make_rope(a, cfg):
    if a.csa:
        return CSA.make_plan(cfg["csa_cfg"], B=1, Tlen=a.seq), \
               CSA.rope_tables(cfg["csa_cfg"], seq_len=a.seq)
    c, s = V.rope_cos_sin(a.seq, a.d // cfg["n_q"], 10000.0)
    return None, (T.tensor(c), T.tensor(s))


# ═════════════════════════════════════════════════════════════════════════
# ORV 参数：命名遍历 / 赋值
# ═════════════════════════════════════════════════════════════════════════
def orv_named(P):
    """(name, tensor) 生成器；名字口径与 `params_of` 的摊平顺序一致。"""
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


def _newt(arr, device=None):
    a = np.ascontiguousarray(np.asarray(arr, np.float32))
    return T.tensor(a, requires_grad=True, device=device)


def orv_set(P, name, arr):
    """按名字把一个 ORV 参数整体替换成 arr（None 表示保持 None）。"""
    parts = name.split(".")
    if parts[0] in ("emb", "ln_f", "head"):
        P[parts[0]] = _newt(arr)
        return
    L = P["layers"][int(parts[1])]
    if parts[2] == "csa":
        L["csa"][parts[3]] = None if arr is None else _newt(arr)
    elif parts[2] in ("W1", "W2", "W3"):
        L[parts[2]][int(parts[3])] = _newt(arr)
    else:
        L[parts[2]] = _newt(arr)


# ═════════════════════════════════════════════════════════════════════════
# 权重映射：torch named_parameters → ORV 名字
# ═════════════════════════════════════════════════════════════════════════
def map_weights(model, cfg_orv, kinds):
    """返回 (assign, keymap)：
    assign[orv_name] = ndarray；keymap[orv_name] = torch 参数名（None 表示 torch 没这个槽位）。
    CSA 子模块复用包里的 `weights_from_torch_state_dict`（conv 方向由它处理）。"""
    sd = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
    csac = cfg_orv["csa_cfg"]
    assign = {"emb": sd["embed.weight"],
              "ln_f": sd["final_norm.weight"],
              "head": sd["lm_head.weight"]}
    keymap = {"emb": "embed.weight", "ln_f": "final_norm.weight", "head": "lm_head.weight"}
    for i, kind in enumerate(kinds):
        p = "blocks.%d." % i
        for src, dst in (("ln1.weight", "ln1"), ("ln2.weight", "ln2"),
                         ("ffn.router.weight", "router")):
            assign["layers.%d.%s" % (i, dst)] = sd[p + src]
            keymap["layers.%d.%s" % (i, dst)] = p + src
        for e in range(cfg_orv["E"]):
            for src, dst in (("w1", "W1"), ("w2", "W2"), ("w3", "W3")):
                tn = "%sffn.experts.%d.%s.weight" % (p, e, src)
                assign["layers.%d.%s.%d" % (i, dst, e)] = sd[tn]
                keymap["layers.%d.%s.%d" % (i, dst, e)] = tn
        for src, dst in (("w1", "S1"), ("w2", "S2"), ("w3", "S3")):
            tn = p + "ffn.shared_experts.0.%s.weight" % src
            assign["layers.%d.%s" % (i, dst)] = sd[tn]
            keymap["layers.%d.%s" % (i, dst)] = tn
        if kind == "csa":
            # 剥掉 `blocks.i.attn.` 前缀，交给包里的映射函数
            sub = {k[len(p + "attn."):]: v for k, v in sd.items() if k.startswith(p + "attn.")}
            w = CSA.weights_from_torch_state_dict(sub, csac)
            for kk, vv in w.items():
                assign["layers.%d.csa.%s" % (i, kk)] = (
                    None if vv is None else
                    np.asarray(vv.to_numpy() if hasattr(vv, "to_numpy") else vv, np.float32))
                keymap["layers.%d.csa.%s" % (i, kk)] = None   # 由该函数内部按原键名取
        else:
            for src, dst in (("q_proj", "wq"), ("k_proj", "wk"),
                             ("v_proj", "wv"), ("o_proj", "wo")):
                tn = p + "attn.%s.weight" % src
                assign["layers.%d.%s" % (i, dst)] = sd[tn]
                keymap["layers.%d.%s" % (i, dst)] = tn
    # CSA 内部的 keymap 用 torch 侧原名补齐（子字典键就是 `q_proj.weight` 这种）
    for i, kind in enumerate(kinds):
        if kind != "csa":
            continue
        p = "blocks.%d.attn." % i
        for src, dst in (("q_proj.weight", "wq"), ("kv_down.weight", "wkv"),
                         ("k_down.weight", "wk"), ("v_down.weight", "wv"),
                         ("o_proj.weight", "wo"), ("conv.weight", "conv"),
                         ("latent_norm.weight", "ln"), ("gate", "gate"),
                         ("q_norm.weight", "qn"), ("k_norm.weight", "kn")):
            keymap["layers.%d.csa.%s" % (i, dst)] = p + src
    return assign, keymap


def apply_and_check(P, assign, model):
    """把 assign 灌进 P，并断言「名字集合 + 形状」与 PyTorch 参数完全一致。"""
    cur = dict(orv_named(P))
    missing = [n for n in assign if n not in cur]
    extra = [n for n, t in cur.items() if t is not None and n not in assign]
    if missing or extra:
        raise AssertionError("ORV 参数名不匹配：缺 %r / 多 %r" % (missing, extra))
    bad = []
    for n, arr in assign.items():
        if arr is None:
            if cur[n] is not None:
                bad.append((n, "torch=None but ORV has tensor"))
            continue
        if cur[n] is None:
            bad.append((n, "torch tensor but ORV None"))
        elif tuple(cur[n].shape) != tuple(np.shape(arr)):
            bad.append((n, "shape ORV=%s torch=%s" % (tuple(cur[n].shape), np.shape(arr))))
    if bad:
        raise AssertionError("ORV/torch 形状不一致：%r" % (bad,))
    for n, arr in assign.items():
        orv_set(P, n, arr)
    n_orv = sum(1 for _, t in orv_named(P) if t is not None)   # None 槽位（qk_norm 关时的 qn/kn）不算
    n_t = len(list(model.named_parameters()))
    if n_orv != n_t:
        raise AssertionError("参数个数不等：ORV %d vs torch %d" % (n_orv, n_t))
    return dict(n_params=n_t, n_tensors=sum(1 for _, t in orv_named(P) if t is not None))


# ═════════════════════════════════════════════════════════════════════════
# 前向（CLI 循环的逐字复刻 + 可选中间量）
# ═════════════════════════════════════════════════════════════════════════
def orv_forward(P, cfg, kinds, plan, cos, sin, ids, trace=None):
    seq, d = len(ids), cfg["d"]
    n_kv_attn = int(cfg.get("n_kv_attn", cfg["n_kv"]))
    x = P["emb"][ids].reshape(1, seq, d)
    if trace is not None:
        trace["embed"] = x
    for Li, L in enumerate(P["layers"]):
        xn = V.rmsnorm_orv(x, L["ln1"])
        if kinds[Li] == "csa":
            attn = CSA.csa_forward(xn, L["csa"], plan, cos, sin)
        else:
            attn = V.attention_orv(xn, L["wq"], L["wk"], L["wv"], L["wo"],
                                   cos, sin, d // cfg["n_q"], cfg["n_q"], n_kv_attn)
        h = x + attn
        hn3 = V.rmsnorm_orv(h, L["ln2"])
        hn = hn3.reshape(seq, d)
        eids, gates, _s = MOE.route_topk(hn, L["router"], int(cfg["top_k"]))
        mo = MOE.moe_topk_orv(hn, eids, gates, L["W1"], L["W2"], L["W3"])
        mo = mo + V.situ_glu_orv(hn, L["S1"], L["S2"], L["S3"])
        if trace is not None:
            trace["L%d.eids" % Li] = eids
        if trace is not None:
            trace["L%d.ln1" % Li] = xn
            trace["L%d.attn" % Li] = attn
            trace["L%d.h" % Li] = h
            trace["L%d.ln2" % Li] = hn3
            trace["L%d.moe" % Li] = mo.reshape(1, seq, d)
        x = h + mo.reshape(1, seq, d)
        if trace is not None:
            trace["L%d.out" % Li] = x
    xf = V.rmsnorm_orv(x, P["ln_f"])
    logits = xf.reshape(seq, d) @ P["head"].transpose(0, 1)
    if trace is not None:
        trace["ln_f"] = xf
        trace["logits"] = logits
    return logits


def orv_loss(logits, ids, style):
    if style == "matched":                       # 与参考实现同口径：丢最后一个位置
        return T.functional.cross_entropy(logits[0:len(ids) - 1], ids[1:])
    tgt = np.roll(ids, -1)                       # CLI 原样口径（含环绕目标）
    return T.functional.cross_entropy(logits, tgt)


def torch_forward_traced(model, ids):
    """逐字复刻 `TGAIV3Model.forward` 的非 AttnRes / 非 ckpt 分支，用于逐层对拍。"""
    inp = torch.tensor(ids[None, :], dtype=torch.long, device="cuda")
    tr = {}
    x = model.embed(inp)
    tr["embed"] = x
    for i, blk in enumerate(model.blocks):
        xn = blk.ln1(x)
        a, _hm = blk.attn(xn, cache=None, use_cache=False, return_attn_max=False)
        h = x + a
        hn3 = blk.ln2(h)
        mo = blk.ffn(hn3)
        tr["L%d.ln1" % i] = xn
        tr["L%d.attn" % i] = a
        tr["L%d.h" % i] = h
        tr["L%d.ln2" % i] = hn3
        tr["L%d.moe" % i] = mo
        x = h + mo
        tr["L%d.out" % i] = x
        ffn = blk.ffn
        if hasattr(ffn, "router"):                     # MoE：记录本侧真实的路由结果
            k = int(model.config.n_activated)
            s = torch.softmax(ffn.router(hn3.reshape(-1, model.config.d_model)).float(), dim=-1)
            tr["L%d.eids" % i] = torch.topk(s, k, dim=-1).indices
    xf = model.final_norm(x)
    logits = model.lm_head(xf)
    tr["ln_f"] = xf
    tr["logits"] = logits
    return logits, tr


# ═════════════════════════════════════════════════════════════════════════
# 度量 / 小工具
# ═════════════════════════════════════════════════════════════════════════
def to_np(x):
    """torch / ORV / numpy 三种张量统一取 numpy（float64 由调用方决定）。"""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy())
    return np.asarray(x)


def to_orv_basis(name, arr, orv_shape):
    """把 torch 参数/梯度换到 ORV 参数空间。

    只有一处需要：CSA 的 depthwise conv。参考实现 `conv.weight` 是 (latent,1,k) 且核方向
    与 ORV 的 (latent,k) **相反**（`csa_conv_causal_orv` 与 `weights_from_torch_state_dict`
    里的 flip 都说明了这点）。权重装载时已经翻过 ⇒ 梯度/权重比较也必须翻一次才是同一
    个基下的量。这是**基变换**，不是放宽容差。
    """
    if name.endswith(".csa.conv"):
        lat, k = orv_shape
        return np.ascontiguousarray(np.flip(np.reshape(arr, (lat, k)), axis=1))
    return arr


def orv_to_torch_basis(name, arr, orv_shape):
    """上式的逆（flip 是对合 ⇒ 再翻一次）。"""
    if name.endswith(".csa.conv"):
        lat, k = orv_shape
        return np.ascontiguousarray(
            np.flip(np.reshape(arr, (lat, k)), axis=1)).reshape(lat, 1, k)
    return arr


def diff(a, b):
    a = np.asarray(to_np(a), np.float64)
    b = np.asarray(to_np(b), np.float64)
    reshaped = False
    if a.shape != b.shape:
        if a.size != b.size:
            return dict(shape_mismatch="%s vs %s" % (a.shape, b.shape), max_abs=float("nan"))
        a, b, reshaped = a.reshape(-1), b.reshape(-1), True
    dmax = float(np.abs(a - b).max()) if a.size else 0.0
    scale = float(np.abs(b).max()) if b.size else 0.0
    return dict(max_abs=dmax, max_rel=dmax / max(scale, 1e-12), ref_scale=scale,
                mean_abs=float(np.abs(a - b).mean()) if a.size else 0.0,
                finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
                reshaped=reshaped)


def opt_state(opt):
    d = getattr(opt, "defaults", None)
    if isinstance(d, dict):                     # torch
        return dict(lr=d["lr"], betas=list(d["betas"]), eps=d["eps"],
                    weight_decay=d["weight_decay"], impl="torch.optim.AdamW")
    return dict(lr=getattr(opt, "lr", None),
                betas=[getattr(opt, "b1", None), getattr(opt, "b2", None)],
                eps=getattr(opt, "eps", None), weight_decay=getattr(opt, "wd", None),
                impl="tgai_go.train.optim.AdamW")


def grad_np(t):
    if t is None:
        return None
    return np.asarray(to_np(t), np.float32)


def snapshot_orv_weights(P):
    return {n: (None if t is None else np.array(t.to_numpy(), copy=True))
            for n, t in orv_named(P)}


def restore_orv_weights(P, snap):
    for n, arr in snap.items():
        orv_set(P, n, arr)


def set_orv_grads(P, grads):
    for n, t in orv_named(P):
        if t is None:
            continue
        g = grads.get(n)
        t.grad = None if g is None else T.tensor(np.ascontiguousarray(g, np.float32))


def scalar(x):
    return float(np.asarray(to_np(x)).reshape(-1)[0])


def orv_version():
    for attr in ("__version__", "VERSION", "_VERSION"):
        v = getattr(T, attr, None)
        if v:
            return str(v)
    try:
        import tgai_go as G
        v = getattr(G, "__version__", None)
        if v:
            return str(v)
    except Exception:
        pass
    try:
        import importlib.metadata as md
        return str(md.version("tgai-orv"))
    except Exception:
        return "unknown"


def norm_of(grads):
    s = 0.0
    for g in grads.values():
        if g is not None:
            s += float((np.asarray(g, np.float64) ** 2).sum())
    return float(np.sqrt(s))


# ═════════════════════════════════════════════════════════════════════════
# selftest：证明"手工前向" == CLI 的训练循环
# ═════════════════════════════════════════════════════════════════════════
def run_selftest(a):
    """用同一 seed 复刻 CLI 的 build→取数据顺序，对比 CLI 自己打印的 step-0 loss。"""
    args = ["--d", str(a.d), "--layers", str(a.layers), "--E", str(a.E),
            "--seq", str(a.seq), "--vocab", str(a.vocab), "--n_q", str(a.n_q),
            "--d_ff", str(a.d_ff), "--top_k", str(a.top_k), "--steps", "1",
            "--lr", str(a.lr)]
    if a.csa:
        args += ["--csa", str(a.csa)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        CLI.main(args)
    text = buf.getvalue()
    m = re.search(r"losses \[([^\]]+)\]", text)      # CLI 原始浮点（不经过 %.4f 打印）
    m2 = re.search(r"step 0 loss=([0-9.eE+-]+)", text)
    if not (m or m2):
        print("[selftest] CLI output:\n%s" % text[-3000:])
        raise RuntimeError("没从 CLI 输出里找到 step 0 loss")
    loss_cli = float(m.group(1).split(",")[0]) if m else float(m2.group(1))
    loss_cli_printed = float(m2.group(1)) if m2 else float("nan")

    # 复刻 CLI：rng = default_rng(0)（CLI 里写死 0）→ build → ids
    aa = argparse.Namespace(**vars(a))
    cfg = orv_cfg(aa, preset_csa_cfg=False)       # 完全照 CLI：由 build() 建 csa_cfg
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    plan, (cos, sin) = make_rope(aa, cfg)
    kinds = CLI.layer_kinds(cfg)
    ids_all = rng.integers(0, cfg["vocab"], (2, cfg["seq"])).astype(np.int64)
    logits = orv_forward(P, cfg, kinds, plan, cos, sin, ids_all[0])
    loss_replica_cli_style = scalar(orv_loss(logits, ids_all[0], "cli"))
    res = dict(loss_cli=loss_cli, loss_cli_printed=loss_cli_printed,
               loss_replica_cli_style=loss_replica_cli_style,
               abs_diff=abs(loss_cli - loss_replica_cli_style),
               text_tail=text.strip().splitlines()[-3:])
    print("[selftest] CLI raw step0 loss=%.9f  replica=%.9f  |d|=%.3e (bit-identical iff 0)"
          % (loss_cli, loss_replica_cli_style, res["abs_diff"]))
    print("[selftest] CLI tail: %r" % (res["text_tail"],))
    return res


# ═════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--n_q", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--csa", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="all",
                    choices=["selftest", "fwd", "grad", "optstep", "opteps",
                             "cross", "curve", "all"])
    ap.add_argument("--bisect", action="store_true", help="逐层打印中间量差异")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    T.set_default_device("cuda")

    out = {"args": vars(a).copy(), "alignment": ALIGNMENT,
           "versions": {"torch": torch.__version__, "orv": orv_version(),
                        "orv_path": getattr(T, "__file__", "?")}}
    t_start = time.time()

    if a.stage == "selftest":
        out["selftest"] = run_selftest(a)
        return _finish(out, a, t_start)

    cfg = orv_cfg(a, qk_norm=False)
    kinds = CLI.layer_kinds(cfg)
    tcfg = torch_cfg(a, qk_norm=False)
    assert kinds == [("global" if tcfg.is_global_layer(i) else "csa")
                     for i in range(a.layers)], "层类型不一致"
    print("[cfg] d=%d layers=%d E=%d seq=%d vocab=%d n_q=%d d_ff=%d top_k=%d csa=%d"
          % (a.d, a.layers, a.E, a.seq, a.vocab, a.n_q, a.d_ff, a.top_k, a.csa))
    print("[cfg] kinds=%r csa_cfg=%r" % (kinds, cfg.get("csa_cfg")))
    out["cfg"] = dict(kinds=kinds, d=a.d, layers=a.layers, E=a.E, seq=a.seq,
                      vocab=a.vocab, n_q=a.n_q, d_ff=a.d_ff, top_k=a.top_k,
                      csa=a.csa, csa_cfg=repr(cfg.get("csa_cfg")))

    # ── ORV 侧：build() 建参数（随后被 torch 权重整体覆盖）──
    rng = np.random.default_rng(a.seed)
    P = CLI.build(cfg, rng)
    plan, (cos, sin) = make_rope(a, cfg)

    # ── PyTorch 侧 ──
    torch.manual_seed(a.seed)
    model = M.TGAIV3Model(tcfg).cuda()
    model.train()
    assign, keymap = map_weights(model, cfg, kinds)
    applied = apply_and_check(P, assign, model)
    out["params"] = applied
    print("[map] %d params mapped 1:1 (name set + shapes asserted), %d tensor slots"
          % (applied["n_params"], applied["n_tensors"]))

    ids_all = np.random.default_rng(a.seed + 1).integers(
        0, a.vocab, (max(a.steps, 1) + 1, a.seq)).astype(np.int64)
    ids = ids_all[0]
    out["warmup_s"] = _warmup(P, cfg, kinds, plan, cos, sin, ids)

    # ── step-0 前向 ──
    tr_o = {}
    logits_o = orv_forward(P, cfg, kinds, plan, cos, sin, ids, trace=tr_o)
    logits_t, tr_t = torch_forward_traced(model, ids)
    inp = torch.tensor(ids[None, :], dtype=torch.long, device="cuda")
    logits_t2, loss_t, _ = model(inp, targets=inp)
    out["replica_check"] = diff(logits_t, logits_t2)
    print("[replica] torch forward-replica vs model.forward: max_abs=%.3e"
          % out["replica_check"]["max_abs"])

    out["forward_logits"] = diff(logits_o, logits_t[0])
    print("[fwd] logits max_abs=%.3e max_rel=%.3e (scale=%.4g)"
          % (out["forward_logits"]["max_abs"], out["forward_logits"]["max_rel"],
             out["forward_logits"]["ref_scale"]))

    loss_o_match = scalar(orv_loss(logits_o, ids, "matched"))
    loss_o_cli = scalar(orv_loss(logits_o, ids, "cli"))
    loss_t_v = scalar(loss_t)
    out["step0_loss"] = dict(
        torch=loss_t_v, orv_matched=loss_o_match, orv_cli_style=loss_o_cli,
        abs_diff=abs(loss_o_match - loss_t_v),
        rel_diff=abs(loss_o_match - loss_t_v) / max(abs(loss_t_v), 1e-12),
        cli_style_abs_diff=abs(loss_o_cli - loss_t_v),
        cli_style_rel_diff=abs(loss_o_cli - loss_t_v) / max(abs(loss_t_v), 1e-12))
    print("[step0] loss torch=%.8f  orv(matched)=%.8f  |d|=%.3e (rel %.2e)  "
          "orv(cli-style)=%.8f |d|=%.3e"
          % (loss_t_v, loss_o_match, out["step0_loss"]["abs_diff"],
             out["step0_loss"]["rel_diff"], loss_o_cli,
             out["step0_loss"]["cli_style_abs_diff"]))

    if a.bisect:
        out["bisect"] = {}
        print("[bisect] stage-by-stage (orv vs torch):")
        for k in [k for k in tr_o if k in tr_t]:
            dn = diff(tr_o[k], tr_t[k])
            out["bisect"][k] = dn
            print("  %-8s max_abs=%.3e max_rel=%.3e scale=%.4g"
                  % (k, dn["max_abs"], dn["max_rel"], dn["ref_scale"]))

    if a.stage == "fwd":
        return _finish(out, a, t_start)

    # ── 反向 + 梯度对拍 ──
    kw = dict(lr=a.lr, betas=tuple(a.betas), eps=a.eps, weight_decay=a.weight_decay)
    opt_o = T.optim.AdamW(CLI.params_of(P), **kw)
    opt_t = torch.optim.AdamW(model.parameters(), **kw)
    out["optimizer"] = dict(torch=opt_state(opt_t), orv=opt_state(opt_o))
    print("[opt] torch=%r" % (out["optimizer"]["torch"],))
    print("[opt] orv  =%r" % (out["optimizer"]["orv"],))

    opt_o.zero_grad()
    loss_o = orv_loss(logits_o, ids, "matched")
    loss_o.backward()
    opt_t.zero_grad(set_to_none=True)
    loss_t.backward()

    g_o = {n: grad_np(t.grad) for n, t in orv_named(P) if t is not None}
    tparams = dict(model.named_parameters())
    g_t = {n: (grad_np(tparams[keymap[n]].grad)
               if keymap.get(n) in tparams else None) for n in g_o}
    if set(g_o) != set(g_t):
        raise AssertionError("梯度名字集合不等：%r" % (set(g_o) ^ set(g_t),))
    rows = []
    orv_shapes = {n: tuple(t.shape) for n, t in orv_named(P) if t is not None}

    def grads_in_orv_basis(name, g):
        return to_orv_basis(name, g, orv_shapes[name])

    def to_torch_space(name, g):
        return orv_to_torch_basis(name, g, orv_shapes[name])

    rev = {tn: on for on, tn in keymap.items() if tn}
    g_t_orv = {n: grads_in_orv_basis(n, g_t[n]) for n in g_t}
    g_o_torch = {n: to_torch_space(n, g_o[n]) for n in g_o}

    for n in g_o:
        if g_o[n] is None or g_t[n] is None:
            rows.append(dict(name=n, missing=True))
            continue
        raw = diff(g_o[n], g_t[n])
        g_cmp = grads_in_orv_basis(n, g_t[n])
        dn = diff(g_o[n], g_cmp)
        dn["name"] = n
        dn["orv_norm"] = float(np.linalg.norm(g_o[n]))
        dn["torch_norm"] = float(np.linalg.norm(g_cmp))
        if n.endswith(".csa.conv"):
            dn["raw_max_rel"] = raw.get("max_rel")
            dn["basis_change"] = "flip kernel axis (same physical operator)"
        rows.append(dn)
    rows.sort(key=lambda r: -r.get("max_rel", -1))
    worst = max(r.get("max_rel", 0.0) for r in rows)
    out["grads"] = dict(worst_rel=worst, worst_10=rows[:10],
                        n_missing=sum(1 for r in rows if r.get("missing")),
                        global_norm_orv=norm_of(g_o), global_norm_torch=norm_of(g_t))
    print("[grad] %d tensors, worst max_rel=%.3e, missing=%d, |g|orv=%.6f |g|torch=%.6f"
          % (len(rows), worst, out["grads"]["n_missing"],
             out["grads"]["global_norm_orv"], out["grads"]["global_norm_torch"]))
    for r in rows[:10]:
        if r.get("missing"):
            print("  %-30s MISSING" % r["name"])
        else:
            print("  %-30s max_abs=%.3e max_rel=%.3e |g|o=%.3e |g|t=%.3e"
                  % (r["name"], r["max_abs"], r["max_rel"], r["orv_norm"], r["torch_norm"]))
    keys = ["emb", "layers.0.ln1", "layers.0.csa.wo", "layers.1.wo",
            "layers.1.W1.0", "head", "layers.0.csa.conv", "layers.0.csa.gate",
            "layers.0.router", "layers.0.csa.ln"]
    out["grads"]["requested"] = {k: next((r for r in rows if r["name"] == k), None)
                                 for k in keys}
    for k in keys:
        r = out["grads"]["requested"][k]
        print("  [req] %-22s %s" % (k, "n/a" if not r else
                                    ("MISSING" if r.get("missing")
                                     else "max_rel=%.3e" % r["max_rel"])))

    if a.stage == "optstep":
        # 决定性隔离：**喂完全相同的梯度**给两个优化器，看一步之后权重差多少。
        #   A：两侧都用 torch 的梯度；B：两侧都用 ORV 的梯度。
        # 若 A/B 里两侧权重仍差 ~1e-4 级 ⇒ 优化器实现语义不同；
        # 若权重一致 ⇒ 差异只能来自梯度（且是 AdamW 对近零梯度噪声的放大）。
        snap_g = snapshot_orv_weights(P)
        snap_tt = {n: v.detach().clone() for n, v in model.named_parameters()}

        def restore_all():
            restore_orv_weights(P, snap_g)
            with torch.no_grad():
                for n, v in model.named_parameters():
                    v.copy_(snap_tt[n])
                    v.grad = None

        def one_step(grads_orv, grads_torch, label):
            restore_all()
            set_orv_grads(P, grads_orv)
            oo = T.optim.AdamW(CLI.params_of(P), **kw)
            oo.step()
            for tn, v in model.named_parameters():
                g = grads_torch.get(rev.get(tn))
                v.grad = None if g is None else torch.tensor(g, dtype=v.dtype, device=v.device)
            torch.optim.AdamW(model.parameters(), **kw).step()
            tpar = dict(model.named_parameters())
            per = {}
            for n, t in orv_named(P):
                if t is None:
                    continue
                wo = np.asarray(to_np(t), np.float64)
                wt = np.asarray(to_orv_basis(n, to_np(tpar[keymap[n]]),
                                             tuple(t.shape)), np.float64)
                per[n] = (float(np.abs(wo - wt).max() / max(np.abs(wt).max(), 1e-12)),
                          float(np.abs(wo - wt).max()), wo - wt, wo, wt)
            rs = sorted(per.items(), key=lambda kv: -kv[1][0])
            return dict(label=label, per=per, worst_rel=rs[0][1][0],
                        top=[(n, v[0], v[1]) for n, v in rs[:6]])

        resA = one_step(g_t_orv, g_t, "same grads = torch's")
        resB = one_step(g_o, g_o_torch, "same grads = ORV's")
        resC = one_step(g_o, g_t, "own grads")
        # AdamW 第一步：m=(1-b1)g, v=(1-b2)g² ⇒ mh=m/bc1=g, vh=v/bc2=g²
        #   ⇒ dw = lr * g / (|g| + eps)
        # 在 |g| ≲ eps 的元素上这是一条**软符号函数**：d(dw)/dg = lr*eps/(|g|+eps)²
        # 在 |g|=eps 时等于 lr/(4eps) = 2.5e4 —— 梯度噪声被放大到权重层面。
        pred = {}
        for n in g_o:
            if g_o[n] is None or g_t[n] is None:
                continue
            gg = np.asarray(g_t_orv[n], np.float64)
            dg = np.asarray(g_o[n], np.float64) - gg
            pred[n] = float((a.lr * np.abs(dg) / (np.abs(gg) + a.eps)).max())
        top_pred = sorted(pred.items(), key=lambda kv: -kv[1])[:6]
        out["optstep"] = dict(
            same_torch_grads=dict(worst_rel=resA["worst_rel"], top=resA["top"]),
            same_orv_grads=dict(worst_rel=resB["worst_rel"], top=resB["top"]),
            own_grads=dict(worst_rel=resC["worst_rel"], top=resC["top"]),
            amplification=dict(k1=a.lr, c2=1.0,
                               formula="dw = lr*g/(|g|+eps) at step 1",
                               d_dw_dg_at_eps=a.lr / (4.0 * a.eps),
                               top=[(n, p, resC["per"].get(n, (0, 0))[1])
                                    for n, p in top_pred],
                               note="predicted |dw diff| from grad diff alone"))
        print("[optstep] feed IDENTICAL grads, one AdamW step, weight rel diff:")
        print("  A same=torch grads: worst_rel=%.3e  %r"
              % (resA["worst_rel"], [(n, float("%.2e" % d)) for n, d, _ in resA["top"]]))
        print("  B same=ORV   grads: worst_rel=%.3e  %r"
              % (resB["worst_rel"], [(n, float("%.2e" % d)) for n, d, _ in resB["top"]]))
        print("  C own grads (what the real run does): worst_rel=%.3e  %r"
              % (resC["worst_rel"], [(n, float("%.2e" % d)) for n, d, _ in resC["top"]]))
        print("[optstep] AdamW step-1 form: dw = lr*g/(|g|+eps)  =>  d(dw)/dg at |g|=eps "
              "is lr/(4*eps)=%.1e" % (a.lr / (4.0 * a.eps)))
        # 元素级验证：在"权重差最大"的那个元素上，用梯度差直接算 AdamW 的更新差
        elem = []
        for n, _rel, _abs in sorted(resC["top"], key=lambda t: -t[2])[:4]:
            dw = resC["per"][n][2]
            idx = np.unravel_index(int(np.abs(dw).argmax()), dw.shape)
            go_i = float(np.asarray(g_o[n], np.float64)[idx])
            gt_i = float(np.asarray(g_t_orv[n], np.float64)[idx])
            w0_o_i = float(np.asarray(to_np(snap_g[n]), np.float64)[idx])
            w0_t_i = float(np.asarray(to_orv_basis(n, to_np(snap_tt[keymap[n]]),
                                                   tuple(np.shape(dw))), np.float64)[idx])
            wo_i = float(resC["per"][n][3][idx])
            wt_i = float(resC["per"][n][4][idx])

            def adamw1(g, w):                     # 与两侧内核逐条同式（第一步）
                m = (1.0 - a.betas[0]) * g
                v = (1.0 - a.betas[1]) * g * g
                mh = m / (1.0 - a.betas[0])
                vh = v / (1.0 - a.betas[1])
                p = w - a.lr * a.weight_decay * w
                return p - a.lr * mh / (math.sqrt(vh) + a.eps)

            elem.append(dict(name=n, index=[int(i) for i in idx],
                             g_orv=go_i, g_torch=gt_i, dg=go_i - gt_i,
                             w0_orv=w0_o_i, w0_torch=w0_t_i, w0_diff=w0_o_i - w0_t_i,
                             dw_orv=wo_i - w0_o_i, dw_torch=wt_i - w0_t_i,
                             dw_orv_pred=adamw1(go_i, w0_o_i) - w0_o_i,
                             dw_torch_pred=adamw1(gt_i, w0_t_i) - w0_t_i,
                             actual_dw=float(dw[idx])))
        out["optstep"]["element_level"] = elem
        print("[optstep] element-level check at the element with max |dw|:")
        for e in elem:
            print("  %-14s idx=%s" % (e["name"], e["index"]))
            print("     g:  orv=%+.4e  torch=%+.4e   (dg=%+.2e)"
                  % (e["g_orv"], e["g_torch"], e["dg"]))
            print("     w0: orv=%.9f torch=%.9f  (diff=%+.2e)"
                  % (e["w0_orv"], e["w0_torch"], e["w0_diff"]))
            print("     dW: orv_act=%+.4e  orv_pred=%+.4e | torch_act=%+.4e  torch_pred=%+.4e"
                  % (e["dw_orv"], e["dw_orv_pred"], e["dw_torch"], e["dw_torch_pred"]))
            print("     dW difference actual=%+.4e" % (e["actual_dw"],))
        for n, p, act in out["optstep"]["amplification"]["top"]:
            print("  [bound] %-18s predicted max|dw|<=%.3e actual=%.3e"
                  % (n, p, act))
        return _finish(out, a, t_start)

    if a.stage == "opteps":
        # 优化器在**不同梯度量级**下的等价性：同一份梯度喂两侧，比 一步后的 Δw。
        # 大批量梯度（~1e-2）里 eps 可忽略；梯度到 1e-9 时 eps=1e-8 主导，
        # 这一步会暴露 eps 的位置/数值口径差异。
        snap = snapshot_orv_weights(P)
        tt = {n: v.detach().clone() for n, v in model.named_parameters()}

        def restore0():
            restore_orv_weights(P, snap)
            with torch.no_grad():
                for n, v in model.named_parameters():
                    v.copy_(tt[n])
                    v.grad = None

        res = {}
        for scale in (1e-3, 1e-6, 1e-9):
            grng = np.random.default_rng(7)
            gs = {}
            for n, t in orv_named(P):
                if t is None:
                    continue
                gs[n] = (grng.normal(0, scale, tuple(t.shape)).astype(np.float32))
            restore0()
            w0_o = {n: np.array(to_np(t), np.float64) for n, t in orv_named(P) if t is not None}
            w0_t = {n: np.array(to_orv_basis(n, to_np(tt[keymap[n]]), tuple(gs[n].shape)),
                                np.float64) for n in gs}
            set_orv_grads(P, gs)
            T.optim.AdamW(CLI.params_of(P), **kw).step()
            tpar = dict(model.named_parameters())
            for n in gs:
                tpar[keymap[n]].grad = torch.tensor(
                    orv_to_torch_basis(n, gs[n], tuple(gs[n].shape)),
                    dtype=tpar[keymap[n]].dtype, device=tpar[keymap[n]].device)
            torch.optim.AdamW(model.parameters(), **kw).step()
            worst = (None, 0.0)
            worst_abs = (None, 0.0)
            now_o = dict(orv_named(P))
            for n in gs:
                wo = np.asarray(to_np(now_o[n]), np.float64) - w0_o[n]
                wt = np.asarray(to_orv_basis(n, to_np(tpar[keymap[n]]),
                                             tuple(gs[n].shape)), np.float64) - w0_t[n]
                den = max(float(np.abs(wt).max()), 1e-30)
                r = float(np.abs(wo - wt).max()) / den
                if r > worst[1]:
                    worst = (n, r)
                a_abs = float(np.abs(wo - wt).max())
                if a_abs > worst_abs[1]:
                    worst_abs = (n, a_abs)
            res["%g" % scale] = dict(worst_param=worst[0], worst_rel=worst[1],
                                     worst_abs_param=worst_abs[0], worst_abs=worst_abs[1])
            print("[opteps] grad scale %.0e: max|dW| = %.3e (%s), worst rel = %.3e (%s)"
                  % (scale, worst_abs[1], worst_abs[0], worst[1], worst[0]))
        restore0()
        out["opteps"] = res
        return _finish(out, a, t_start)

    if a.stage == "grad":
        return _finish(out, a, t_start)

    # ── 交叉喂优化器（决定性隔离实验）──
    snap_o = snapshot_orv_weights(P)
    snap_t = {n: v.detach().clone() for n, v in model.named_parameters()}
    ids_next = ids_all[1]

    def torch_step_loss(grads_torch_space):
        """键是 **ORV 名字**（便于两侧用同一套名字对照），值是 torch 参数形状的梯度。"""
        for tn, v in model.named_parameters():
            g = grads_torch_space.get(rev.get(tn))
            v.grad = None if g is None else torch.tensor(g, dtype=v.dtype, device=v.device)
        o = torch.optim.AdamW(model.parameters(), **kw)
        o.step()
        with torch.no_grad():
            nxt = torch.tensor(ids_next[None, :], dtype=torch.long, device="cuda")
            _lg, ls, _ = model(nxt, targets=nxt)
        return scalar(ls)

    def torch_restore():
        with torch.no_grad():
            for n, v in model.named_parameters():
                v.copy_(snap_t[n])
                v.grad = None

    def orv_step_loss(grads):
        restore_orv_weights(P, snap_o)
        o = T.optim.AdamW(CLI.params_of(P), **kw)
        set_orv_grads(P, grads)
        o.step()
        lg = orv_forward(P, cfg, kinds, plan, cos, sin, ids_next)
        return scalar(orv_loss(lg, ids_next, "matched"))

    torch_restore()
    L_tt = torch_step_loss(g_t)          # g_t 本身就在 torch 参数空间
    torch_restore()
    L_tO = torch_step_loss(g_o_torch)
    torch_restore()
    L_OO = orv_step_loss(g_o)
    L_Ot = orv_step_loss(g_t_orv)
    restore_orv_weights(P, snap_o)
    out["cross"] = dict(
        torch_own=L_tt, torch_with_orv_grad=L_tO, orv_own=L_OO, orv_with_torch_grad=L_Ot,
        d_torch_grad_swap=abs(L_tt - L_tO), d_orv_grad_swap=abs(L_OO - L_Ot),
        d_framework=abs(L_tO - L_OO),
        rel=dict(d_torch_grad_swap=abs(L_tt - L_tO) / max(abs(L_tt), 1e-12),
                 d_orv_grad_swap=abs(L_OO - L_Ot) / max(abs(L_OO), 1e-12),
                 d_framework=abs(L_tO - L_OO) / max(abs(L_tO), 1e-12)))
    print("[cross] torch AdamW: own grads=%.8f  ORV grads=%.8f  |d|=%.3e"
          % (L_tt, L_tO, abs(L_tt - L_tO)))
    print("[cross] ORV   AdamW: own grads=%.8f  torch grads=%.8f  |d|=%.3e"
          % (L_OO, L_Ot, abs(L_OO - L_Ot)))
    print("[cross] cross-framework next loss: torch(orv grads)=%.8f vs orv(own)=%.8f |d|=%.3e"
          % (L_tO, L_OO, abs(L_tO - L_OO)))

    if a.stage == "cross":
        return _finish(out, a, t_start)

    # ── N 步曲线 ──
    # 三组对照：
    #   (a) 训练损失：每步用**不同**的随机批（与 CLI 一致）
    #   (b) 评估损失：每步用**同一批** held-out ids（平滑的"loss 下降曲线"）
    #   (c) 敏感性对照：PyTorch 从 w0 出发，但先加一个相对 ~1e-7 的扰动，再跑同样的 20 步。
    #       若 ORV-vs-torch 的偏离与 (c) 同量级 ⇒ 偏离就是浮点噪声被动力学放大，
    #       不是数学差异（这是把"曲线差"归因的关键证据）。
    restore_orv_weights(P, snap_o)
    torch_restore()
    opt_o = T.optim.AdamW(CLI.params_of(P), **kw)
    opt_t = torch.optim.AdamW(model.parameters(), **kw)
    eval_ids = ids_all[-1]
    tparams = dict(model.named_parameters())

    def rel_dist():
        num = den = 0.0
        for n, t in orv_named(P):
            if t is None:
                continue
            wo = np.asarray(to_np(t), np.float64)
            wt = np.asarray(to_orv_basis(n, to_np(tparams[keymap[n]]),
                                         tuple(t.shape)), np.float64)
            if wo.shape != wt.shape:
                raise AssertionError("rel_dist 形状不一致 %s: %s vs %s"
                                     % (n, wo.shape, wt.shape))
            num += float(((wo - wt) ** 2).sum())
            den += float((wt ** 2).sum())
        return float(np.sqrt(num / max(den, 1e-30)))

    def orv_eval():
        return scalar(orv_loss(orv_forward(P, cfg, kinds, plan, cos, sin, eval_ids),
                               eval_ids, "matched"))

    def torch_eval(m):
        inp_e = torch.tensor(eval_ids[None, :], dtype=torch.long, device="cuda")
        with torch.no_grad():
            _lg, ls, _ = m(inp_e, targets=inp_e)
        return scalar(ls)

    curve = {"torch": [], "orv": [], "eval_torch": [], "eval_orv": [],
             "grad_norm_torch": [], "grad_norm_orv": [], "weight_rel_dist": [],
             "route_mismatch": [], "top_param_diff": []}
    d0 = rel_dist()          # 起点自检：必须是 0（同一份权重）
    if d0 > 1e-12:
        raise AssertionError("起点权重不一致：rel_dist=%.3e" % d0)
    print("[curve] start weight rel_dist=%.3e (must be 0)" % d0)
    t_ids = torch.tensor(ids_all[None, :], dtype=torch.long, device="cuda")
    tV = tcfg.vocab_size
    for step in range(a.steps):
        ids_s = ids_all[step]
        tr_o = {}
        lo_s = orv_forward(P, cfg, kinds, plan, cos, sin, ids_s, trace=tr_o)
        loss_o_s = orv_loss(lo_s, ids_s, "matched")
        opt_o.zero_grad()
        loss_o_s.backward()
        if step < 3:
            curve["grad_norm_orv"].append(norm_of(
                {n: grad_np(t.grad) for n, t in orv_named(P) if t is not None}))
        opt_o.step()
        curve["orv"].append(scalar(loss_o_s))

        logits_ts, tr_ts = torch_forward_traced(model, ids_s)
        tgt = torch.tensor(ids_s[None, 1:], dtype=torch.long, device="cuda")
        loss_t_s = torch.nn.functional.cross_entropy(
            logits_ts[:, :-1].contiguous().view(-1, tV), tgt.reshape(-1), ignore_index=-100)
        opt_t.zero_grad(set_to_none=True)
        loss_t_s.backward()
        if step < 3:
            curve["grad_norm_torch"].append(norm_of(
                {n: grad_np(v.grad) for n, v in model.named_parameters()}))
        opt_t.step()
        curve["torch"].append(scalar(loss_t_s))

        # 路由一致性：每层有多少 token 的专家集合不同（离散量，不是浮点噪声）
        mm = {}
        for i in range(a.layers):
            e_o = np.sort(np.asarray(tr_o["L%d.eids" % i]).reshape(len(ids_s), -1), axis=1)
            e_t = np.sort(tr_ts["L%d.eids" % i].cpu().numpy().reshape(len(ids_s), -1), axis=1)
            mm[i] = dict(n_rows_differ=int((e_o != e_t).any(axis=1).sum()),
                         n_tokens=len(ids_s),
                         top1_differ=int((e_o[:, 0] != e_t[:, 0]).sum()))
        curve["route_mismatch"].append(mm)

        curve["eval_orv"].append(orv_eval())
        curve["eval_torch"].append(torch_eval(model))
        curve["weight_rel_dist"].append(rel_dist())
        if step < 4:
            rs = []
            for n, t in orv_named(P):
                if t is None:
                    continue
                wo = np.asarray(to_np(t), np.float64)
                wt = np.asarray(to_orv_basis(n, to_np(tparams[keymap[n]]),
                                             tuple(t.shape)), np.float64)
                rs.append((n, float(np.abs(wo - wt).max() / max(np.abs(wt).max(), 1e-12))))
            rs.sort(key=lambda kv: -kv[1])
            curve["top_param_diff"].append(rs[:5])

    # (c) 敏感性对照：PyTorch 从同一份 w0 出发，先加一个相对扰动，再跑同样的 20 步。
    #     跑两个幅度：(i) 实测"1 步后"的权重差；(ii) 实测"20 步后"的权重差。
    #     若 ORV-vs-torch 的损失差与 (ii) 同量级 ⇒ 观测到的损失差就是那点权重差的必然结果。
    def run_control(eps):
        m2 = M.TGAIV3Model(tcfg).cuda()
        m2.train()
        with torch.no_grad():
            for n, v in m2.named_parameters():
                v.copy_(snap_t[n])
        gen = torch.Generator(device="cuda").manual_seed(12345)
        with torch.no_grad():
            for v in m2.parameters():
                nz = torch.randn(v.shape, generator=gen, device=v.device)
                v.add_(nz * (eps * v.abs() + 1e-30))
        opt2 = torch.optim.AdamW(m2.parameters(), **kw)
        tr_l, ev_l = [], []
        for step in range(a.steps):
            inp_s = t_ids[:, step, :]
            _lg, l2, _ = m2(inp_s, targets=inp_s)
            opt2.zero_grad(set_to_none=True)
            l2.backward()
            opt2.step()
            tr_l.append(scalar(l2))
            ev_l.append(torch_eval(m2))
        del m2
        return tr_l, ev_l

    eps_early = curve["weight_rel_dist"][0]
    eps_final = curve["weight_rel_dist"][-1]
    ctrl_early = run_control(eps_early)
    ctrl_final = run_control(eps_final)

    dt = np.abs(np.array(curve["torch"]) - np.array(curve["orv"]))
    de = np.abs(np.array(curve["eval_torch"]) - np.array(curve["eval_orv"]))
    dc_e = np.abs(np.array(curve["eval_torch"]) - np.array(ctrl_early[1]))
    dc_f = np.abs(np.array(curve["eval_torch"]) - np.array(ctrl_final[1]))
    out["curve"] = dict(
        steps=a.steps, torch=curve["torch"], orv=curve["orv"],
        abs_diff=dt.tolist(), max_abs_diff=float(dt.max()), max_rel_diff=float(de.max()),
        final_ratio=float(curve["orv"][-1] / curve["torch"][-1]),
        last_torch=float(curve["torch"][-1]), last_orv=float(curve["orv"][-1]),
        eval_torch=curve["eval_torch"], eval_orv=curve["eval_orv"],
        eval_abs_diff=de.tolist(), eval_max_abs=float(de.max()),
        eval_final_ratio=float(curve["eval_orv"][-1] / curve["eval_torch"][-1]),
        weight_rel_dist=curve["weight_rel_dist"], weight_rel_dist_init=d0,
        route_mismatch=curve["route_mismatch"], top_param_diff=curve["top_param_diff"],
        monotone_torch=dict(n_decreases=sum(1 for i in range(1, len(curve["eval_torch"]))
                                           if curve["eval_torch"][i] < curve["eval_torch"][i - 1]),
                            n_steps=len(curve["eval_torch"]) - 1),
        monotone_orv=dict(n_decreases=sum(1 for i in range(1, len(curve["eval_orv"]))
                                         if curve["eval_orv"][i] < curve["eval_orv"][i - 1]),
                          n_steps=len(curve["eval_orv"]) - 1),
        grad_norm_torch=curve["grad_norm_torch"], grad_norm_orv=curve["grad_norm_orv"],
        control=dict(eps_early=eps_early, eps_final=eps_final,
                     eval_early=ctrl_early[1], eval_final=ctrl_final[1],
                     eval_max_abs_early=float(dc_e.max()),
                     eval_max_abs_final=float(dc_f.max()),
                     ratio_vs_orv_early=float(de.max() / max(float(dc_e.max()), 1e-30)),
                     ratio_vs_orv_final=float(de.max() / max(float(dc_f.max()), 1e-30))))
    print("[curve] step   train_torch   train_orv    |d|t      eval_torch    eval_orv     |d|e")
    for i in range(a.steps):
        print("  %2d   %.8f  %.8f  %.2e  %.8f  %.8f  %.2e"
              % (i, curve["torch"][i], curve["orv"][i], dt[i],
                 curve["eval_torch"][i], curve["eval_orv"][i], de[i]))
    print("[curve] train-loss max|d|=%.3e   eval-loss max|d|=%.3e  final eval ratio=%.8f"
          % (dt.max(), de.max(), out["curve"]["eval_final_ratio"]))
    print("[curve] weight rel-dist per step: %r"
          % [float("%.3e" % x) for x in curve["weight_rel_dist"]])
    print("[curve] eval curve monotone (n_decreases/steps): torch=%r orv=%r"
          % (out["curve"]["monotone_torch"], out["curve"]["monotone_orv"]))
    print("[curve] route mismatch (rows differ / top1 differ, per layer):")
    for k, mm in enumerate(curve["route_mismatch"]):
        print("  step %2d  %r" % (k, {i: (v["n_rows_differ"], v["top1_differ"])
                                      for i, v in mm.items()}))
    print("[curve] top-5 per-param rel diff after steps 0..3:")
    for k, rs in enumerate(curve["top_param_diff"]):
        print("  step %d: %r" % (k, [(n, float("%.3e" % d)) for n, d in rs]))
    print("[curve] CONTROL torch+perturb(rel=%.2e): eval max|d| vs torch=%.3e "
          "(ORV was %.3e; ratio orv/control=%.2f)"
          % (eps_early, dc_e.max(), de.max(), out["curve"]["control"]["ratio_vs_orv_early"]))
    print("[curve] CONTROL torch+perturb(rel=%.2e, = final drift): eval max|d|=%.3e "
          "(ORV was %.3e; ratio orv/control=%.2f)"
          % (eps_final, dc_f.max(), de.max(), out["curve"]["control"]["ratio_vs_orv_final"]))
    print("[curve] grad_norm step0-2 torch=%r orv=%r"
          % ([round(x, 8) for x in curve["grad_norm_torch"]],
             [round(x, 8) for x in curve["grad_norm_orv"]]))
    return _finish(out, a, t_start)


def _warmup(P, cfg, kinds, plan, cos, sin, ids):
    """首次调用含 NVRTC JIT（实测约 5 s），单独计时并从对拍结果里剔除。"""
    t0 = time.time()
    lg = orv_forward(P, cfg, kinds, plan, cos, sin, ids)
    _ = scalar(orv_loss(lg, ids, "matched"))
    dt = time.time() - t0
    print("[warmup] first ORV forward %.2f s (includes NVRTC JIT; excluded from results)" % dt)
    return dt


def _finish(out, a, t_start):
    out["elapsed_s"] = time.time() - t_start
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("[json] written %s" % a.json)
    print("=== JSON ===")
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
