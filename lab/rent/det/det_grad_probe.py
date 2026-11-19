# -*- coding: utf-8 -*-
"""梯度级定位：同一进程里做两次 forward+backward，逐参数比较梯度是否逐位相同。

用途：`TG_DETERMINISTIC=1` 仍不可复现时，找出**是哪一类参数的梯度**在变
（embedding / router / experts / csa / ln / head），从而定位剩下的原子散射。

跑法：D:\\orv_bench\\remote_py.bat D:\\orv_bench\\det_grad_probe.py D:\\orv_det_work\\new_pkg_x
"""
import json
import os
import sys

PKG = sys.argv[1] if len(sys.argv) > 1 else ""
DET = len(sys.argv) > 2 and sys.argv[2].lower() in ("1", "det", "true", "on")
if PKG:
    sys.path.insert(0, PKG)

import numpy as np                                              # noqa: E402

import tgai_go.train as T                                       # noqa: E402
from tgai_go.train import v3_pretrain as VP                     # noqa: E402
from tgai_go.train import v3_train_cli as CLI                   # noqa: E402

OUT = r"D:\orv_bench\det_grad_probe.json"
ROOT = r"D:\orv_bench"


def build(a):
    T.set_default_device("cuda")
    cfg = VP._expand_cfg(a)
    kinds = CLI.layer_kinds(cfg)
    n_csa = sum(1 for k in kinds if k == "csa")
    if n_csa:
        from tgai_go.train import v3_csa as _C
        cfg["_csa_mod"] = _C
    rng = np.random.default_rng(42)
    P = CLI.build(cfg, rng)                     # csa>0 时它会写回 cfg["csa_cfg"]
    if n_csa:
        cos, sin = cfg["_csa_mod"].rope_tables(cfg["csa_cfg"], seq_len=a.seq)
    else:
        cos, sin = T.tensor(np.zeros(1, np.float32)), T.tensor(np.zeros(1, np.float32))
    params = CLI.params_of(P)
    names = VP.param_names(P)
    return cfg, kinds, cos, sin, P, params, names


def grads_of(params, names):
    out = {}
    for nm, t in zip(names, params):
        g = getattr(t, "grad", None)
        if g is None:
            out[nm] = None
        else:
            out[nm] = np.asarray(g.to_numpy(), np.float32).copy()
    return out


def main():
    argv = ["--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
            "--seq", "2048", "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "adamw", "--amp", "gemm", "--batch", "1", "--steps", "1",
            "--log-every", "0", "--no-final-save"]
    a = VP.build_arg_parser().parse_args(argv)
    cfg, kinds, cos, sin, P, params, names = build(a)
    corpus = VP.PackedCorpus(VP.expand_data_paths(a.data), seq=a.seq, seed=42,
                             vocab=a.vocab)
    opt = T.optim.AdamW(params, lr=1e-3)
    ids, labels, seg = corpus.take(1)
    rt = T._cuda()
    if DET:
        rt.deterministic = True
    print("[probe] deterministic=%r  ids.shape=%s  n_unique_ids=%d"
          % (rt.deterministic, ids.shape, len(np.unique(ids))), flush=True)
    recs = []
    from collections import Counter
    kcount = Counter()
    _orig_launch = rt.launch

    def _counting(name, *aa, **kw):
        kcount[name] += 1
        return _orig_launch(name, *aa, **kw)

    rt.launch = _counting
    for rep in range(2):
        opt.zero_grad()
        loss, meta = VP.forward_loss(P, cfg, kinds, cos, sin, ids, labels, seg,
                                     top_k=2, shared=True, use_seg=True)
        lv = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
        loss.backward()
        rt.sync()
        recs.append({"loss": lv, "grads": grads_of(params, names)})
        print("[probe] rep%d loss=%r" % (rep, lv), flush=True)
        if rep == 0:
            print("[probe] launch counts (rep0): %s"
                  % (dict(sorted(kcount.items())),), flush=True)
            kcount.clear()
    res = {"loss_equal": recs[0]["loss"] == recs[1]["loss"],
           "loss0": recs[0]["loss"], "loss1": recs[1]["loss"], "params": []}
    for nm in names:
        g0, g1 = recs[0]["grads"][nm], recs[1]["grads"][nm]
        if g0 is None or g1 is None:
            res["params"].append({"name": nm, "grad": None})
            continue
        neq = int(np.count_nonzero(g0 != g1))
        res["params"].append({
            "name": nm, "n": int(g0.size), "n_diff": neq,
            "max_abs_diff": float(np.abs(g0.astype(np.float64)
                                         - g1.astype(np.float64)).max()) if neq else 0.0,
        })
    bad = [p for p in res["params"] if p.get("n_diff")]
    res["n_params_differing"] = len(bad)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[probe] loss_equal=%s  params_differing=%d/%d"
          % (res["loss_equal"], len(bad), len(names)), flush=True)
    for p in sorted(bad, key=lambda q: -q["n_diff"])[:40]:
        print("   %-46s diff %7d/%7d  max|d| %.3e"
              % (p["name"], p["n_diff"], p["n"], p["max_abs_diff"]), flush=True)
    if not bad:
        print("   （无差异参数）", flush=True)
    print("[probe] wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
