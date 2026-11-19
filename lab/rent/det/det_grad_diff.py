# -*- coding: utf-8 -*-
"""数值对照：同一批数据、同一步，`TG_DETERMINISTIC` 开 vs 关的梯度差异。

期望：**接近但不完全相等** —— 求和顺序变了（跨行原子累加 → 按行号升序串行），
末几位会不同，相对误差应在 fp32 舍入量级（1e-7~1e-6），不应该是 1e-3 这种量级。

跑法：D:\\orv_bench\\remote_py.bat D:\\orv_bench\\det_grad_diff.py D:\\orv_det_work\\new_pkg_x
"""
import json
import os
import sys

PKG = sys.argv[1] if len(sys.argv) > 1 else ""
if PKG:
    sys.path.insert(0, PKG)

import numpy as np                                              # noqa: E402

import tgai_go.train as T                                       # noqa: E402
from tgai_go.train import v3_pretrain as VP                     # noqa: E402
from tgai_go.train import v3_train_cli as CLI                   # noqa: E402

OUT = r"D:\orv_bench\det_grad_diff.json"
ROOT = r"D:\orv_bench"


def main():
    argv = ["--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
            "--seq", "2048", "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "adamw", "--amp", "gemm", "--batch", "1", "--steps", "1"]
    a = VP.build_arg_parser().parse_args(argv)
    T.set_default_device("cuda")
    cfg = VP._expand_cfg(a)
    kinds = CLI.layer_kinds(cfg)
    from tgai_go.train import v3_csa as C
    cfg["_csa_mod"] = C
    P = CLI.build(cfg, np.random.default_rng(42))
    params = CLI.params_of(P)
    names = VP.param_names(P)
    cos, sin = C.rope_tables(cfg["csa_cfg"], seq_len=a.seq)
    corpus = VP.PackedCorpus(VP.expand_data_paths(a.data), seq=a.seq, seed=42,
                             vocab=a.vocab)
    ids, labels, seg = corpus.take(1)
    rt = T._cuda()
    opt = T.optim.AdamW(params, lr=1e-3)

    def run(det):
        rt.deterministic = bool(det)
        opt.zero_grad()
        loss, _ = VP.forward_loss(P, cfg, kinds, cos, sin, ids, labels, seg,
                                  top_k=2, shared=True, use_seg=True)
        lv = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
        loss.backward()
        rt.sync()
        g = {}
        for nm, t in zip(names, params):
            if getattr(t, "grad", None) is not None:
                g[nm] = np.asarray(t.grad.to_numpy(), np.float64).copy()
        return lv, g

    lv_off, g_off = run(False)
    lv_on, g_on = run(True)
    res = {"loss_off": lv_off, "loss_on": lv_on, "loss_equal": lv_off == lv_on,
           "n_params": len(names), "per_param": []}
    num = 0.0
    den = 0.0
    for nm in names:
        if nm not in g_off:
            continue
        a_ = g_off[nm]
        b_ = g_on[nm]
        num += float(np.sum((b_ - a_) ** 2))
        den += float(np.sum(a_ ** 2))
        md = float(np.abs(b_ - a_).max())
        neq = int(np.count_nonzero(a_ != b_))
        if neq:
            res["per_param"].append({"name": nm, "n": int(a_.size), "n_diff": neq,
                                     "max_abs_diff": md})
    res["global_rel_err"] = (num / den) ** 0.5 if den > 0 else 0.0
    res["n_params_differing"] = len(res["per_param"])
    res["gnorm_off"] = float(np.sqrt(den))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("loss off=%r on=%r equal=%s" % (lv_off, lv_on, res["loss_equal"]))
    print("grad: params differing %d/%d, global rel err = %.3e, ||g||=%.4f"
          % (res["n_params_differing"], len(names), res["global_rel_err"],
             res["gnorm_off"]))
    for p in sorted(res["per_param"], key=lambda q: -q["max_abs_diff"])[:12]:
        print("   %-42s diff %6d/%6d  max|d| %.3e"
              % (p["name"], p["n_diff"], p["n"], p["max_abs_diff"]))
    print("[det_grad_diff] wrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
