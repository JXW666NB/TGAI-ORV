# -*- coding: utf-8 -*-
"""把 ORV 迷你 V3 模型的权重 + 数据块导出成 npz，供 PyTorch 侧逐位复现。"""
import json
import os
import sys

import numpy as np

import pbench_lib as L

L.utf8()
import tgai_go.train as T

assert T.get_device() == "cpu"
T.manual_seed(1000)
model = T.preset.transformer(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512)
ps = list(model.parameters())
names, arrays = [], {}
seen = set()
for n, p in model.named_parameters():
    a = np.asarray(p.data, np.float32)
    if id(p) in seen:
        continue
    seen.add(id(p))
    names.append(n)
    arrays[n] = a
print("params:", names)

xs, ys = L.model_blocks(64, 260)
X = np.stack(xs).astype(np.int64)
Y = np.stack(ys).astype(np.int64)

blob = {"X": X, "Y": Y, "names": np.array(names, dtype=object)}
for n, a in arrays.items():
    blob["w::" + n] = a

out = os.path.join(L.out_dir(), "state_v4096.npz")
np.savez(out, **blob)
print("WROTE " + out + "  bytes=%d" % os.path.getsize(out))

# 记录 ORV 侧的初始 loss（同一实现，供 torch 侧对齐）
lg = model(X[:8])
l0 = L.ce_vec(T, lg.reshape(8 * 64, 4096), Y[:8].reshape(-1))
meta = {"names": names, "shapes": {n: list(map(int, arrays[n].shape)) for n in names},
        "init_loss_b8": float(np.asarray(l0.to_numpy())),
        "n_params": int(model.n_params()),
        "cfg": {"vocab": 4096, "d_model": 128, "n_layers": 2, "n_heads": 4, "d_ff": 512}}
with open(os.path.join(L.out_dir(), "state_v4096_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print(json.dumps(meta, ensure_ascii=True, indent=1))
