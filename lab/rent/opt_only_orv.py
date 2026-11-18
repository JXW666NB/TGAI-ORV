# -*- coding: utf-8 -*-
"""单独测优化器一步的耗时（O(参数量) 的固定开销）。

为什么要测：每一步都有与 token 数无关的开销（参数更新、梯度裁剪、动量缓冲读写）。
seq 越大，这笔开销被摊得越薄 —— 这正是"PyTorch 每 token 成本随 seq 下降"的主因候选。
两边优化器精度不同（ORV m8v16 vs PyT fp32 AdamW），所以固定开销本来就不同量级。
"""
import json
import sys
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()

LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
DENSE = (len(sys.argv) > 2 and sys.argv[2] == "dense")
DFF = 13312 if DENSE else 3328
E = 1 if DENSE else 4
TK = 1 if DENSE else 2

cfg = dict(d=1024, d_ff=DFF, E=E, layers=LAYERS, n_q=16, n_kv=2, vocab=96000,
           seq=512, attn_res=False, attn_res_block=2, top_k=TK, csa=4,
           n_kv_attn=16, rope_theta=500000.0, csa_compress=4,
           csa_conv_kernel=3, csa_local_window=0)
P = CLI.build(cfg, np.random.default_rng(0))
params = CLI.params_of(P)
n_par = sum(int(np.prod(tuple(int(s) for s in p.shape))) for p in params)

# 给所有参数塞一份非零梯度，模拟反向之后的状态
for p in params:
    p.grad = T.tensor(np.full(int(np.prod(tuple(int(s) for s in p.shape))),
                              1e-3, np.float32))

# 与 v3_pretrain 用同一个优化器类（int8gpu = Int8AdamWGpu，m8v16）
from tgai_go.train.int8_adamw_gpu import Int8AdamWGpu
opt = Int8AdamWGpu(params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                   weight_decay=0.1)
clip = T.clip_grad_norm_

ts = []
for i in range(4):
    rt.sync()
    t0 = time.perf_counter()
    if clip is not None:
        try:
            clip(params, 1.0)
        except TypeError:
            clip(params)
    opt.step()
    rt.sync()
    dt = time.perf_counter() - t0
    if i >= 1:
        ts.append(dt)
med = sorted(ts)[len(ts) // 2]
print(json.dumps({"impl": "orv", "layers": LAYERS, "dense": DENSE,
                  "params": n_par, "opt_step_s": med}))
