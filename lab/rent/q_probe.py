# -*- coding: utf-8 -*-
"""探针：远端数据分片形状/取值区间 + ORV 包版本。只读，不做训练。"""
import glob
import os
import sys

import numpy as np

DATA = r"D:\orv_bench\data2048_v96000"
paths = sorted(glob.glob(os.path.join(DATA, "*.npz")))
print("shards:", len(paths))
tot_rows = 0
for p in paths:
    with np.load(p) as z:
        ids = z["input_ids"]
        lab = z["labels"] if "labels" in z.files else None
        seg = z["seg_id"] if "seg_id" in z.files else None
        print("%s ids=%s %s  lab=%s  seg=%s  id[%d,%d]  lab==ids %.4f  sup %.4f"
              % (os.path.basename(p), ids.shape, ids.dtype,
                 None if lab is None else (lab.shape, lab.dtype),
                 None if seg is None else (seg.shape, seg.dtype),
                 int(ids.min()), int(ids.max()),
                 float((np.asarray(lab) == ids).mean()) if lab is not None else -1.0,
                 float((np.asarray(lab) != -100).mean()) if lab is not None else -1.0))
        tot_rows += int(ids.shape[0])
print("total rows:", tot_rows, " windows@512:", tot_rows * (2048 // 512))

# ORV 侧的参考实现（窗口顺序必须逐位复刻）
sys.path.insert(0, r"D:\orv_pip_0962\Lib\site-packages")
import tgai_go  # noqa: E402
from tgai_go.train import v3_pretrain as VP  # noqa: E402

print("tgai_go version:", getattr(tgai_go, "__version__", "?"))
infos = [VP.probe_shard(p, 512) for p in paths]
print("probe_shard:", [(os.path.basename(i.path), i.n_rows, i.width, i.n_windows)
                       for i in infos])
perm = np.random.default_rng(42).permutation(len(paths))
print("shard_order(epoch0):", perm.tolist())
si = int(perm[0])
w = np.random.default_rng(int(42 * 1000003 + si * 7919)).permutation(infos[si].n_windows)
print("first 8 window idx of shard %d:" % si, w[:8].tolist())

import torch  # noqa: E402
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
