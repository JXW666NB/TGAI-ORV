# -*- coding: utf-8 -*-
"""训练检查点：**权重（v4）+ 优化器状态（fp16 动量）+ 元信息**，支持轮转与续训。

设计要点（都对应实测约束）：

* **权重写成 v4 `.TG`**：与产线同容器，训练产物可直接被引擎加载（实测 GPU 接管）。
  写盘走 `tg4.write_tg4_stream`（单张量内存，**TG4 容器**），分词段从源 `.tg` 原样复用（`tok_blob`）。
* **动量写成每层一个 `npz`**：10.9 GB 不能整体驻留内存；`save_opt_states` 逐层落盘。
* **`meta.json` 记录 step / loss / 时间 / 张量数**：续训需要步数，排查需要上下文。
* **轮转**：完整检查点 16.96 GB（权重 6.06 + 动量 10.90），磁盘有限，只留最近 `keep` 个。

用法见 `tools/v27_train_run.py`。
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import struct
import time

import numpy as np

CKPT_WEIGHTS = "weights.tg"
CKPT_OPT_DIR = "opt"
CKPT_META = "meta.json"


def read_tok_blob(tg_path):
    """取 `.TG`（v3/v4）的分词段原始字节，用于原样复用到新文件。"""
    with open(tg_path, "rb") as f:
        head = f.read(128)
        magic = head[:4]
        if magic != b"TGAI":
            raise ValueError("不是 TGAI 容器：%s（magic=%r）。"
                             "修法：检查点权重必须是 .TG（v3 或 TG4；本版本起新检查点写 TG4），"
                             "先确认路径没指错、文件没被别的格式覆盖" % (tg_path, magic))
        tok_off, tok_len = struct.unpack_from("<2Q", head, 64)
        f.seek(tok_off)
        return f.read(tok_len)


def names_sidecar_path(tg_path):
    """找到可用的 `*.names.json`，返回路径或 None。

    回退顺序（实测：v4 文件没有自己的边车，但张量表与 v3 完全相同）：
      ① 同名 `<tg>.names.json`
      ② 去掉版本后缀的兄弟，如 `X_v4.tg` → `X.tg.names.json`
      ③ 同目录唯一一个 `*.names.json`（唯一才用，多个就拒绝，避免张冠李戴）
    """
    exact = tg_path + ".names.json"
    if os.path.exists(exact):
        return exact
    d, base = os.path.dirname(os.path.abspath(tg_path)), os.path.basename(tg_path)
    stem = base
    for suf in (".tg", ".TG"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    # 去掉 _v4 / -v4 之类的版本尾巴
    import re
    stem2 = re.sub(r"[_-]v\d+$", "", stem)
    if stem2 != stem:
        cand = os.path.join(d, stem2 + ".tg.names.json")
        if os.path.exists(cand):
            return cand
    cands = sorted(glob.glob(os.path.join(d, "*.names.json")))
    # 只保留"看起来属于同一模型族"的（公共前缀 >= 6 字符）
    same_family = [c for c in cands
                   if os.path.basename(c)[:6].lower() == base[:6].lower()]
    if len(same_family) == 1:
        return same_family[0]
    return None


def read_names_sidecar(tg_path):
    """读张量名表。找不到返回 None（调用方应跳过检查点并给出修法提示）。"""
    p = names_sidecar_path(tg_path)
    if not p:
        return None
    with open(p, encoding="utf-8") as f:
        return [t["name"] for t in json.load(f)["tensors"]]


def iter_named_tensors(tr, st, names):
    """按 `.TG` 的张量名顺序，从训练器取出**当前**权重（fp32 ndarray）。

    训练器内部是拆分后的模块（q/k/v 分头、out_proj 分头、专家分散），
    这里做反向拼装；拼装方向必须与导出/加载侧一致（实测：q/k/v 走 axis=0，out_proj 走 axis=1）。
    """
    for n in names:
        if n == "token_embedding.weight":
            yield n, np.asarray(tr.head.weight.to_numpy(), np.float32)
        elif n == "final_norm.weight":
            yield n, np.asarray(tr.final_norm.weight.to_numpy(), np.float32)
        elif ".attn." in n and n.endswith("_proj.weight"):
            i = int(n.split(".")[1])
            kind = n.split(".")[-2].split("_")[0]
            if kind in ("q", "k", "v"):
                yield n, np.concatenate([np.asarray(m.weight.to_numpy(), np.float32)
                                          for m in tr.layers[i][kind]], axis=0)
            else:
                yield n, np.concatenate([np.asarray(m.weight.to_numpy(), np.float32)
                                          for m in tr.layers[i]["o"]], axis=1)
        elif ".moe.router.weight" in n:
            i = int(n.split(".")[1])
            yield n, np.asarray(tr.layers[i]["moe"].router.weight.to_numpy(), np.float32)
        elif ".moe.experts." in n:
            i = int(n.split(".")[1])
            e = int(n.split(".")[4])
            slot = {"w1": "gate", "w2": "up", "w3": "down"}[n.split(".")[-2]]
            yield n, np.asarray(getattr(tr.layers[i]["moe"].experts[e], slot).weight.to_numpy(), np.float32)
        elif n.endswith("moe._qb_bias"):
            i = int(n.split(".")[1])
            yield n, np.asarray(tr.layers[i]["moe"].qb_bias.to_numpy(), np.float32)
        elif n.endswith(("ln1.weight", "ln2.weight")):
            i = int(n.split(".")[1])
            key = "ln1" if n.endswith("ln1.weight") else "ln2"
            yield n, np.asarray(tr.layers[i][key].weight.to_numpy(), np.float32)
        else:
            # 其余（如 rope.cos/sin）训练中不改，直接从源文件取
            yield n, np.asarray(st[n], np.float32)


def save_checkpoint(ckpt_dir, tr, st, opts, head_opt, names, src_tg, step, cfg, meta=None):
    """写一个完整检查点，返回目录路径。"""
    from tgai_go.train.opt_fp16mv import save_opt_states

    os.makedirs(ckpt_dir, exist_ok=True)
    tok_blob = read_tok_blob(src_tg)
    # 头部是定长整型字段：这里统一强转，避免 float/None 混进来
    # （实测：rope_theta 传 float 会让 struct.pack 报 "required argument is not an integer"）
    c = {k: int(cfg[k]) for k in ("n_layers", "d_model", "n_heads", "d_ff", "vocab",
                                  "max_seq", "n_experts", "n_activated", "rope_theta")
         if k in cfg}
    c.update({k: int(cfg.get(k) or 0) for k in ("pad", "bos", "eos", "unk",
                                                "n_kv_heads", "arch")})
    cfg_out = dict(cfg)
    cfg_out.update(c)
    wpath = os.path.join(ckpt_dir, CKPT_WEIGHTS)
    # ★ 2026-11-13：检查点权重改写 **TG4** 容器（自描述分区 + NAM 名字表 + 内置分词表）。
    #
    # 为什么换：v3 家族（含"v3 布局 + version=4"）的权重表只有 FNV 哈希、没有名字，
    # 做不了优化器映射 / 部分加载 / 按名 diff；而 TG4 的 NAM 直接给出名字。
    # 文件名**保持 `weights.tg`** —— 路径是外部契约，容器由 magic 区分：
    # `read_tg()` 两种都认（TG4 走 `_read_tg4` 翻成同构 info），所以旧检查点仍可读取/续训。
    from tgai_go import tg4
    cfg4 = dict(n_layers=c["n_layers"], d_model=c["d_model"], n_heads=c["n_heads"],
                head_dim=int(cfg.get("head_dim") or (c["d_model"] // max(1, c["n_heads"]))),
                d_ff=c["d_ff"], vocab=c["vocab"], max_seq=c["max_seq"],
                n_kv_heads=c.get("n_kv_heads", 0), n_experts=c["n_experts"],
                n_activated=c["n_activated"], arch=c.get("arch", 0),
                weight_dtype=int(tg4.DT_BF16), rope_theta=c["rope_theta"],
                pad_id=c["pad"], bos_id=c["bos"], eos_id=c["eos"], unk_id=c["unk"])
    tg4.write_tg4_stream(wpath, iter_named_tensors(tr, st, names), cfg4, tok=tok_blob)

    odir = os.path.join(ckpt_dir, CKPT_OPT_DIR)
    os.makedirs(odir, exist_ok=True)
    save_opt_states(opts, odir)
    if head_opt is not None:
        import numpy as np2
        sd = head_opt.state_dict()
        np2.savez(os.path.join(odir, "head.npz"), t=np2.int64(sd["t"]),
                  **{("m_%d" % k): a for k, a in enumerate(sd["m"])},
                  **{("v_%d" % k): a for k, a in enumerate(sd["v"])})

    info = {"step": int(step), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tensors": len(names), "weights_bytes": os.path.getsize(wpath),
            "version": 4, "cfg": cfg_out}
    if meta:
        info.update(meta)
    with open(os.path.join(ckpt_dir, CKPT_META), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return ckpt_dir


def load_checkpoint(ckpt_dir, opts, head_opt):
    """恢复优化器状态，返回 meta（含 step）。权重由调用方用 `TGState` 从 `weights.tg` 装载。"""
    from tgai_go.train.opt_fp16mv import load_opt_states

    meta = {}
    mp = os.path.join(ckpt_dir, CKPT_META)
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    odir = os.path.join(ckpt_dir, CKPT_OPT_DIR)
    load_opt_states(opts, odir)
    hp = os.path.join(odir, "head.npz")
    if head_opt is not None and os.path.exists(hp):
        z = np.load(hp, allow_pickle=False)
        n = len(head_opt.params)
        head_opt.load_state_dict({"t": int(z["t"]),
                                  "m": [z["m_%d" % k] for k in range(n)],
                                  "v": [z["v_%d" % k] for k in range(n)]})
    return meta


def list_checkpoints(root):
    """列出 root 下所有含 meta.json 的检查点目录，按 step 升序。"""
    out = []
    for d in glob.glob(os.path.join(root, "*")):
        mp = os.path.join(d, CKPT_META)
        if os.path.isdir(d) and os.path.exists(mp):
            try:
                with open(mp, encoding="utf-8") as f:
                    out.append((int(json.load(f).get("step", -1)), d))
            except Exception:
                continue
    out.sort()
    return out


def rotate(root, keep=2):
    """只保留最近 `keep` 个检查点，删除更早的。返回被删目录列表。"""
    cks = list_checkpoints(root)
    if keep <= 0 or len(cks) <= keep:
        return []
    removed = []
    for _step, d in cks[:-keep]:
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d)
    return removed


def checkpoint_dir(root, step):
    return os.path.join(root, "step_%06d" % int(step))
