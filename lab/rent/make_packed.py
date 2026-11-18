# -*- coding: utf-8 -*-
"""在租来的机器上**直接生成** packed 分片（省掉上传几 GB 语料的时间）。

格式与真 V3 语料一致（`v3_pretrain.probe_shard` 认的那种）：
    <out>/shard_000.npz  内含 input_ids.npy 形状 (N, seq) ＋ labels.npy（同为 (N,seq)）

用途：**只用于测性能**（形状、dtype、分片结构和真实一致）。
⚠️ token 是随机数 ⇒ loss 没有语义意义（会停在 ln(vocab) 附近），别用它判断收敛。
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出目录（会自动创建）")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=96000)
    ap.add_argument("--shards", type=int, default=16, help="分片数")
    ap.add_argument("--rows", type=int, default=64, help="每个分片多少行 (N)")
    ap.add_argument("--dtype", default="int32", choices=["int32", "int64", "uint16"])
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dt = np.dtype(a.dtype)
    if dt == np.uint16 and a.vocab > 65535:
        raise SystemExit("uint16 装不下 vocab=%d" % a.vocab)
    rng = np.random.default_rng(a.seed)
    total = 0
    for i in range(a.shards):
        ids = rng.integers(0, a.vocab, size=(a.rows, a.seq)).astype(dt)
        # labels = 行内右移一位（跨行不接续；对性能测量无影响）
        lab = np.concatenate([ids[:, 1:], ids[:, -1:]], axis=1)
        path = os.path.join(a.out, "shard_%03d.npz" % i)
        np.savez_compressed(path, input_ids=ids, labels=lab)
        total += ids.size
    print("生成 %d 个分片：%s" % (a.shards, a.out))
    print("  每片 (%d, %d)，共 %d 个 token，dtype=%s"
          % (a.rows, a.seq, total, a.dtype))
    print("  典型大小：%s" % os.path.join(a.out, "shard_000.npz"))


if __name__ == "__main__":
    main()
