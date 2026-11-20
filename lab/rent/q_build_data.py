# -*- coding: utf-8 -*-
"""用**权威的 `data_v3.py::BestFitPacker`** 造一份**真文本**的 packed 分片。

## 为什么必须自己造

`D:\orv_bench\data2048_v96000` 是 `make_packed.py` 生成的**随机 token**（其 docstring 自己写着
"只用于测性能 …… 别用它判断收敛"），而且：
  * `labels` 是**行内右移一位**（不是 V3 的 `labels[t]==ids[t]`）⇒ 两侧都会再多错一位；
  * **没有 `seg_id`** ⇒ 块对角掩码 / CSA 分段池化这两条路径根本没被走到
    —— 而 CSA 掩码 bug 修的正是这条路径。
⇒ 用随机数据做的"训练质量对比"是无效的：loss 只会停在 ln(96000) 附近。

## 本脚本做什么

读 `cot_clean.jsonl` 的真中文，按**字节**映射到 id `256+b`（落在 96000 词表内，
只用 256 行；这样不需要 tokenizer，且 next-byte 预测有真实可学结构），
再交给**权威 `data_v3.BestFitPacker`** 打包成 `(N, seq)` 的
`input_ids / labels / seg_id` —— labels 的 -100 规则、seg 的文档号、pad 的 seg=-1
全部由参考实现自己决定，我一个字节都不改。

## 口径提示（必须写进报告）

* id = 256 + utf8_byte ⇒ 词表 96000 里只有 256 行被用到；两侧完全相同。
* `seg_id` 真实存在 ⇒ 块对角掩码与 CSA 分段池化**都被真实覆盖**。
* 分片宽度 = --seq（默认 2048），这样 `--seq 512` 时每行切 4 个窗口，与 ORV 的切法一致。
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np

REF = os.environ.get("HOPE_REF_DIR", r"D:\orv_bench\ref15b")
sys.path.insert(0, REF)
import data_v3 as D  # noqa: E402  权威打包器


def iter_text(paths, limit_chars):
    n = 0
    for p in paths:
        with io.open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    t = o.get("text") if isinstance(o, dict) else None
                except Exception:
                    t = None
                if not isinstance(t, str):
                    t = line
                if len(t) < 32:
                    continue
                yield t
                n += len(t)
                if n >= limit_chars:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\orv_bench\cot_clean.jsonl")
    ap.add_argument("--out", default=r"D:\orv_bench\data2048_real")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--chars", type=int, default=1_200_000)
    ap.add_argument("--piece", type=int, default=403,
                    help="把每篇文本再切成这么多**字节**的片段当一篇『文档』；"
                         "太小则一行挤很多篇，太大则一行只有一篇（seg 退化）。"
                         "403 时 2048 一行约装 5 篇 ⇒ 块对角掩码/CSA 分段真被覆盖")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    packer = D.BestFitPacker(a.seq, n_bins=64, eos_id=259, bos_id=258, pad_id=256)

    rows = []
    n_docs = 0
    for text in iter_text([a.src], a.chars):
        raw = text.encode("utf-8")
        for i in range(0, len(raw), a.piece):
            piece = [256 + b for b in raw[i:i + a.piece]]
            if len(piece) < 32:
                continue
            for r in packer.add(piece):
                rows.append(r)
            n_docs += 1
        for r in packer.flush_ready():
            rows.append(r)
        if len(rows) >= a.shards * a.rows:
            break
    for r in packer.flush_all():
        rows.append(r)

    print("[build] docs=%d rows=%d chars<=%d" % (n_docs, len(rows), a.chars))
    need = a.shards * a.rows
    while len(rows) < need:                     # 不足就复制（loss 对比不需要多样性）
        rows.append(rows[len(rows) % max(1, len(rows))])
    rows = rows[:need]

    for s in range(a.shards):
        chunk = rows[s * a.rows:(s + 1) * a.rows]
        ids = np.stack([r[0] for r in chunk]).astype(np.int32)
        lab = np.stack([r[1] for r in chunk]).astype(np.int32)
        seg = np.stack([r[2] for r in chunk]).astype(np.int16)
        p = os.path.join(a.out, "shard_%03d.npz" % s)
        np.savez_compressed(p, input_ids=ids, labels=lab, seg_id=seg)
        print("  %s  ids=%s lab==ids %.4f  sup %.4f  seg[%d,%d] nseg/row~%.1f"
              % (os.path.basename(p), ids.shape,
                 float((lab == ids).mean()), float((lab != -100).mean()),
                 int(seg.min()), int(seg.max()), float((seg >= 0).sum(-1).mean() / a.seq)),
              flush=True)
    tot = a.shards * a.rows * a.seq
    print("[build] total %d tokens (=%d windows @512)" % (tot, tot // 512))


if __name__ == "__main__":
    raise SystemExit(main())
