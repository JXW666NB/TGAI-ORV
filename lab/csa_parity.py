# -*- coding: utf-8 -*-
"""CSA 数值对拍：ORV 的 v3_csa.csa_forward  vs  PyTorch 的 CompressedSparseAttention。

在 V100 上跑（需要 torch + ORV 0.9.83 同进程）：
    C:\\Python311\\python.exe _csa_parity.py
前提：本目录下有 model_v3.py 与 v3_csa.py。
"""
import sys

sys.path.insert(0, r"D:\orv_pip_0962\Lib\site-packages")   # ORV 0.9.83
sys.path.insert(0, r"D:\TGAI-V3-1.5B\03_src")              # model_v3.py（真项目）

import numpy as np
import torch

import model_v3 as M
import v3_csa as C
import tgai_go.train as T

T.set_default_device("cuda")

D, NH, SEQ, B = 64, 4, 32, 1
cfg_t = M.TGAIV3Config(vocab_size=4096, d_model=D, n_layers=1, n_heads=NH,
                       n_kv_heads=0, d_ff=128, n_experts=2, n_activated=1,
                       max_seq_len=64, csa_compress=4, csa_conv_kernel=3,
                       csa_local_window=0, csa_latent=0)
mod = M.CompressedSparseAttention(cfg_t).cuda().eval()

cfg_o = C.CSAConfig(d_model=D, n_heads=NH, n_kv_heads=0, csa_compress=4,
                    csa_latent=0, csa_conv_kernel=3, csa_local_window=0)
sd = {k: v.detach().cpu().numpy() for k, v in mod.state_dict().items()}
w = C.weights_from_torch_state_dict(sd, cfg_o)

rng = np.random.default_rng(0)
x_n = rng.normal(0, 1, (B, SEQ, D)).astype(np.float32)

with torch.no_grad():
    y_t = mod(torch.tensor(x_n, device="cuda"))
    y_t = y_t[0] if isinstance(y_t, tuple) else y_t
    y_t = y_t.cpu().numpy()

plan = C.make_plan(cfg_o, B, SEQ)
cos, sin = C.rope_tables(cfg_o, SEQ)
y_o = C.csa_forward(T.tensor(x_n), w, plan, cos, sin).to_numpy()

if y_o.ndim == 3 and y_t.ndim == 2:
    y_t = y_t.reshape(y_o.shape)
if y_o.shape != y_t.shape:
    print("CSA_SHAPE_MISMATCH orv=%s torch=%s" % (y_o.shape, y_t.shape))
else:
    d = np.abs(y_o - y_t)
    scale = max(float(np.abs(y_t).max()), 1e-9)
    print("CSA_PARITY max_abs=%.3e max_rel=%.3e scale=%.4f"
          % (float(d.max()), float(d.max()) / scale, scale))
