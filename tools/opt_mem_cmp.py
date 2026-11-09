# -*- coding: utf-8 -*-
"""优化器状态显存对比（每个进程只建一种优化器，避免池回收干扰读数）。

用法（分两次跑）：
    python opt_mem_cmp.py --opt adamw
    python opt_mem_cmp.py --opt adafactor

输出：参数形状推出的状态字节数（公式）+ GPU 实测增量（nvidia-smi）。
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
NL = 28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", choices=["adamw", "adafactor"], required=True)
    args = ap.parse_args()

    import tgai_go.train as T
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW
    from tgai_go.train.v27_train import TGState, V27Trainer

    cfg = dict(d_model=1024, n_heads=16, d_ff=4096, n_experts=8, n_activated=2,
               n_layers=32, rope_theta=10000.0)
    # 真实头信息
    from tgai_go.read_tg import close_tg, read_tg
    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    cfg = dict(d_model=int(h["d_model"]), n_heads=int(h["n_heads"]), d_ff=int(h["d_ff"]),
               n_experts=int(h["n_experts"]), n_activated=int(h["n_activated"]),
               n_layers=int(h["n_layers"]), rope_theta=float(h["rope_theta"]))

    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    tr.attach_head(st)

    # 统计参数形状（真实值）
    shapes = [tuple(int(s) for s in p.shape) for i in range(NL) for p in tr.params_of(i)]
    shapes += [tuple(int(s) for s in p.shape) for p in tr.head_params()]
    adam_b = sum(2 * int(np.prod(s)) * 2 for s in shapes)          # fp16 m/v
    ada_b = 0
    for s in shapes:
        if len(s) >= 2:
            ada_b += (s[-1] + int(np.prod(s[:-1]))) * 4
        else:
            ada_b += int(np.prod(s)) * 4
    want = adam_b if args.opt == "adamw" else ada_b
    log("参数 %d 个 | 公式状态量 %.3f GB" % (len(shapes), want / 1024 ** 3))

    base = RUN.vram()
    if args.opt == "adamw":
        objs = [Fp16MomAdamW(tr.params_of(i), lr=1e-3) for i in range(NL)]
        objs.append(Fp16MomAdamW(tr.head_params(), lr=1e-3))
    else:
        objs = [T.optim.Adafactor(tr.params_of(i), lr=1e-3) for i in range(NL)]
        objs.append(T.optim.Adafactor(tr.head_params(), lr=1e-3))
    after = RUN.vram()
    log("GPU: 建前 %.0f MiB → 建后 %.0f MiB，增量 %.3f GB（公式 %.3f GB）"
        % (base, after, (after - base) / 1024.0, want / 1024 ** 3))
    st.close()
    T._cuda().empty_cache()


if __name__ == "__main__":
    main()
