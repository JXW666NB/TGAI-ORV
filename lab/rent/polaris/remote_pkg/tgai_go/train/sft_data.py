# -*- coding: utf-8 -*-
"""SFT 数据加载：从语料库 jsonl 读出样本 → BPE 分词 → 打包成定长批次。

为什么要单独一层：
* 正式训练必须吃**多条不同样本**（单样本重复只会记忆化，loss 曲线没有意义）；
* 训练用 v4 的 BPE（`tgai_go.bpe`），与产线同一套词表，不能另造；
* 打包要定长（`seq`），不足补齐、超出截断，并给出 **loss 掩码**（补齐位置不计损失）。

用法：
    ds = SftDataset("D:\\SFT语料库", seq=512)
    for ids_in, ids_tgt, mask in ds.batches(batch_size=1, max_batches=3):
        ...
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

SFT_ROOT = "D:\\SFT\u8bed\u6599\u5e93"


def iter_jsonl_texts(root=SFT_ROOT, min_chars=80, limit=None):
    """按文件名排序遍历，产出 (文件名, text)。"""
    n = 0
    for p in sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)):
        base = os.path.basename(p)
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        t = (json.loads(line).get("text") or "").strip()
                    except Exception:
                        continue
                    if len(t) < min_chars:
                        continue
                    yield base, t
                    n += 1
                    if limit is not None and n >= limit:
                        return
        except Exception:
            continue


class SftDataset(object):
    def __init__(self, tokenizer, seq=512, root=SFT_ROOT, min_chars=80):
        self.tk = tokenizer
        self.seq = int(seq)
        self.root = root
        self.min_chars = int(min_chars)

    def encode_one(self, text):
        """文本 → (ids_in, ids_tgt, mask)，定长 `seq`。

        mask 为 1 的位置参与 loss；补齐位置为 0（不计损失），最后一个位置没有 target 也置 0。
        """
        ids = np.asarray(self.tk.encode(text, add_bos=True), np.int64)
        if ids.size < 2:
            return None
        n = min(ids.size - 1, self.seq - 1)
        x = np.zeros(self.seq, np.int64)
        y = np.zeros(self.seq, np.int64)
        m = np.zeros(self.seq, np.float32)
        x[:n] = ids[:n]
        y[:n] = ids[1:n + 1]
        m[:n] = 1.0
        return x, y, m

    def batches(self, batch_size=1, max_batches=None, limit=None):
        """产出 (x, y, mask)，形状均为 [B, seq]。"""
        B = int(batch_size)
        xs, ys, ms = [], [], []
        nb = 0
        for _name, text in iter_jsonl_texts(self.root, self.min_chars, limit):
            enc = self.encode_one(text)
            if enc is None:
                continue
            xs.append(enc[0]); ys.append(enc[1]); ms.append(enc[2])
            if len(xs) == B:
                yield (np.stack(xs), np.stack(ys), np.stack(ms))
                nb += 1
                xs, ys, ms = [], [], []
                if max_batches is not None and nb >= max_batches:
                    return
        if xs:
            yield (np.stack(xs), np.stack(ys), np.stack(ms))
