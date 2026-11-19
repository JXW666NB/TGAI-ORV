# -*- coding: utf-8 -*-
"""数据探针：v4096 分片到底是"有结构"的语料还是随机 id？"""
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

out = {}
ROOT = r"D:\orv_bench"


def stats(path, do_cond=True):
    f = np.load(path)
    ids = np.asarray(f["input_ids"])
    lab = np.asarray(f["labels"])
    flat = ids.reshape(-1).astype(np.int64)
    V = int(flat.max()) + 1
    r = {"shape": list(ids.shape), "dtype": str(ids.dtype), "V": V}
    r["labels_equal_shift"] = bool(np.array_equal(lab[:, :-1], ids[:, 1:]))
    r["n_unique"] = int(np.unique(flat).size)
    cnt = np.bincount(flat, minlength=V).astype(np.float64)
    p = cnt / cnt.sum()
    p = p[p > 0]
    r["unigram_entropy_bits"] = float(-(p * np.log2(p)).sum())
    r["log2_vocab_upper"] = float(np.log2(V))
    same = float((flat[1:] == flat[:-1]).mean())
    r["adjacent_repeat_rate"] = same
    r["expected_if_uniform"] = 1.0 / V
    r["first20"] = [int(v) for v in flat[:20]]
    if do_cond:
        n = min(200000, flat.size - 1)
        a, b = flat[:n], flat[1:n + 1]
        joint = np.zeros((V, V), np.float64)
        np.add.at(joint, (a, b), 1.0)
        pa = joint.sum(axis=1, keepdims=True)
        pb_a = np.divide(joint, pa, out=np.zeros_like(joint), where=pa > 0)
        nz = pb_a > 0
        h = float(-(joint[nz] / n * np.log2(pb_a[nz])).sum())
        r["cond_entropy_bits"] = h
        r["mutual_info_bits"] = r["unigram_entropy_bits"] - h
    return r


for name in ("data2048_v4096", "data2048_v96000", "data2048"):
    d = os.path.join(ROOT, name)
    if not os.path.isdir(d):
        out[name] = "MISSING"
        continue
    files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    out[name] = {"n_files": len(files),
                 "shard0": stats(os.path.join(d, files[0]),
                                 do_cond=(name == "data2048_v4096")) if files else None}

# 语料样本（看是否像真文本）
try:
    p = os.path.join(ROOT, "cot_clean.jsonl")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = [f.readline() for _ in range(2)]
        out["cot_clean_head"] = [ln[:300] for ln in lines]
except Exception as e:
    out["cot_clean_err"] = str(e)

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(d, exist_ok=True)
p = os.path.join(d, "p05_data.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)
print("WROTE " + p)
print(json.dumps(out, ensure_ascii=True, indent=1, default=str))
