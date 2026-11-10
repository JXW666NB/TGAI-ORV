# -*- coding: utf-8 -*-
"""GPU 上验证高层训练器：DPO 一步 + 多步收敛；GRPO 一步。

运行（需独占 GPU）：python rl_trainer_gpu.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log


def main():
    import tgai_go.train as T
    from tgai_go.train.rl_trainer import DPOTrainer, GRPOTrainer

    T.set_default_device("cuda")
    rt = T._cuda()
    dim = 6
    rng = np.random.default_rng(0)
    w = T.nn.Parameter(rng.normal(0, 0.2, (dim,)).astype(np.float32))

    def logprob(ids, mask=None):
        return (T.tensor(np.asarray(ids, np.float32)) * w).sum()

    xw = np.array([1.0, 0.8, 0.2, 0.0, 0.0, 0.1], np.float32)
    xl = np.array([0.0, 0.0, 0.1, 1.0, 0.9, 0.0], np.float32)

    opt = T.optim.AdamW([w], lr=0.2)
    tr = DPOTrainer([w], logprob, beta=0.1, optimizer=opt, max_grad_norm=1.0)
    first = tr.step(xw, None, xl, None, ref_logp_w=0.0, ref_logp_l=0.0)
    for _ in range(20):
        last = tr.step(xw, None, xl, None, ref_logp_w=0.0, ref_logp_l=0.0)
    rt.sync()
    log("DPO(GPU)：first loss %.6f margin %.4f → last loss %.6f margin %.4f（裁剪范数 %.3f）"
        % (first["loss"], first["margin"], last["loss"], last["margin"],
           last.get("clipped_norm") or 0.0))
    ok = last["loss"] < first["loss"] and last["margin"] > first["margin"]
    log("DPO 收敛：%s" % ("PASS" if ok else "FAIL"))

    # GRPO：G=4 条，token 级 logπ
    G, TT = 4, 5
    ids = [rng.normal(0, 1, (TT, dim)).astype(np.float32) for _ in range(G)]
    masks = [np.ones(TT, np.float32) for _ in range(G)]
    rewards = np.array([1.0, 0.5, -0.2, -0.9], np.float32)

    def tok(ids_i, mask_i):
        return (T.tensor(np.asarray(ids_i, np.float32)) * w).sum(axis=1)

    gt = GRPOTrainer([w], tok, clip_eps=0.2, beta=0.0, optimizer=T.optim.AdamW([w], lr=0.05),
                     max_grad_norm=1.0)
    out = gt.step(ids, masks, rewards)
    rt.sync()
    log("GRPO(GPU)：loss %.6f  adv %s  裁剪范数 %.3f"
        % (out["loss"], np.round(out["adv"], 3).tolist(), out.get("clipped_norm") or 0.0))
    finite = np.isfinite(out["loss"])
    log("GRPO 数值有限：%s" % ("PASS" if finite else "FAIL"))

    T.set_default_device("cpu")
    log("总结论：%s" % ("PASS" if (ok and finite) else "FAIL"))
    return 0 if (ok and finite) else 1


if __name__ == "__main__":
    sys.exit(main())
