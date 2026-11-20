# -*- coding: utf-8 -*-
"""HOPE V3 参照侧最小训练 harness（**只读复用 `_src15b` 的权威源码**，不改它）。

为什么不用 `_src15b/train_v3.py`：那是生产脚本（要语料目录 + 一堆配置 + muon/fp16
+ 梯度累积 + 存档），不适合做"逐步 loss 对拍"。本文件只做一件事：
**用权威的 `model_v3.py` 建模型、按 ORV `PackedCorpus` 逐位复刻的窗口顺序喂同一份分片、
AdamW + 梯度裁剪 + 恒定 lr 跑 N 步、逐步打印 loss**，输出格式与 ORV 的 `losses` 对齐。

## 两种模式

* `--mode match`（默认，**主对比**）：把参考模型的架构开关调到 ORV `v3_pretrain`
  实际实现的那一套，使"两侧 = 同一个模型"：
      attn_res_enabled=False      （ORV v3_pretrain 不建 AttnRes 参数）
      moe_post_norm=False         （ORV v3_pretrain 不建 post_norm）
      moe_balance='sign' + 不调 update_moe_bias（ORV 侧 bias 恒 0）
      不加 aux_loss              （ORV v3_pretrain 明确未实现，见其 docstring 第 71-73 行）
      qk_clip_tau=0              （ORV v3_pretrain 未实现 QK-Clip）
      Global 层的 q_norm/k_norm 换成 Identity（ORV 的 attention_orv 没有 QK-Norm；
                                  CSA 层的 qk_norm 两侧都有 ⇒ 保留）
* `--mode full`：参考实现的**全部默认**（AttnRes + post_norm + quantile QB + aux loss
  + QK-Clip）—— 用来量化"ORV 缺的那几项到底值多少 loss"。

## 口径

* 数据：`np.load` 的 `input_ids/labels/seg_id`，窗口顺序**逐位复刻**
  `v3_pretrain.PackedCorpus`（shard_order=default_rng(seed+epoch*100003).permutation,
  window_order=default_rng(seed*1000003+epoch*100003+si*7919).permutation）。
* 损失：`CE(logits[:, :-1], labels[:, 1:], ignore_index=-100)`（参考实现内建，与 ORV 同式）。
  只报**纯 CE**（HOPE V3 的生产日志也是纯 CE，aux 只进反传）。
* 优化器：AdamW(lr, betas=(0.9,0.999), eps=1e-8, wd=0.1)；无 lr 调度（恒定）；
  `clip_grad_norm_(1.0)`（ORV 默认 --grad-clip 1.0）。
* AMP：`torch.amp.autocast('cuda', fp16)`，**不加 GradScaler**（ORV 的 AMP 也没有
  loss scaling —— 它只把 GEMM 输入 cast 成 fp16）。`--amp none` 则全程 fp32。
* 梯度检查点：**关**（两侧都关）。
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

REF = os.environ.get("HOPE_REF_DIR", r"D:\orv_bench\ref15b")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="分片目录（顶层 *.npz）")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2816)
    ap.add_argument("--vocab", type=int, default=96000)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--csa", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42, help="数据窗口顺序用的 seed（ORV 默认 42）")
    ap.add_argument("--init-seed", type=int, default=1234, help="权重初始化 seed（独立于数据序）")
    ap.add_argument("--amp", default="fp16", choices=["fp16", "none"])
    ap.add_argument("--mode", default="match", choices=["match", "full"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tag", default="pyt")
    ap.add_argument("--json-out", default=None)
    return ap.parse_args(argv)


# ══════════════════════════════════════════════════════════════════════
# 数据：逐位复刻 ORV v3_pretrain.PackedCorpus 的窗口顺序
# ══════════════════════════════════════════════════════════════════════
def probe_paths(data):
    paths = []
    for ext in ("*.npz", "*.npy", "*.bin"):
        paths.extend(glob.glob(os.path.join(data, ext)))
    paths = sorted(p for p in paths if os.path.isfile(p))
    if not paths:
        raise SystemExit("--data 没匹配到分片：%r" % (data,))
    infos = []
    for p in paths:
        with np.load(p) as z:
            shp = z["input_ids"].shape
            keys = list(z.files)
        n_rows, width = int(shp[0]), int(shp[1])
        infos.append(dict(path=p, n_rows=n_rows, width=width,
                          n_windows=n_rows * (width // 1), keys=keys))
    return paths, infos


class WindowStream(object):
    """ORV `PackedCorpus` 的窗口顺序（batch=1、单进程、从 epoch0/rank0 开始）。"""

    def __init__(self, data, seq, seed=42):
        self.paths, infos = probe_paths(data)
        self.seq = int(seq)
        self.seed = int(seed)
        self.infos = []
        for i in infos:
            nw = i["n_rows"] * (i["width"] // self.seq)
            self.infos.append(dict(i, n_windows=nw))
        self.n_shards = len(self.infos)
        self._cache = {}
        self.epoch = 0
        self.rank = 0
        self.offset = 0
        self.taken = 0
        self._shard_orders = {}
        self._orders = {}
        print("[data] %d shards  seq=%d  windows=%d  keys=%s"
              % (self.n_shards, self.seq,
                 sum(i["n_windows"] for i in self.infos),
                 self.infos[0]["keys"]), flush=True)

    def _load(self, si):
        got = self._cache.get(si)
        if got is not None:
            return got
        info = self.infos[si]
        with np.load(info["path"]) as z:
            ids = np.asarray(z["input_ids"])
            lab = np.asarray(z["labels"]) if "labels" in z.files else None
            seg = np.asarray(z["seg_id"]) if "seg_id" in z.files else None
        got = (ids, lab, seg)
        if len(self._cache) >= 2:
            self._cache.pop(next(iter(self._cache)))
        self._cache[si] = got
        return got

    def _shard_perm(self, epoch):
        got = self._shard_orders.get(int(epoch))
        if got is None:
            got = np.random.default_rng(self.seed + int(epoch) * 100003).permutation(self.n_shards)
            self._shard_orders[int(epoch)] = got
        return got

    def _win_perm(self, epoch, rank):
        key = (int(epoch), int(rank))
        got = self._orders.get(key)
        if got is None:
            si = int(self._shard_perm(epoch)[rank])
            mix = (self.seed * 1000003 + int(epoch) * 100003 + si * 7919) % (2 ** 62)
            got = np.random.default_rng(int(mix)).permutation(self.infos[si]["n_windows"])
            self._orders[key] = got
        return got

    def next_window(self):
        """逐行照 `PackedCorpus._next()`（batch=1）。"""
        while True:
            if self.rank >= self.n_shards:
                self.epoch += 1
                self.rank = 0
                self.offset = 0
            si = int(self._shard_perm(self.epoch)[self.rank])
            n = self.infos[si]["n_windows"]
            if n <= 0 or self.offset >= n:
                self.rank += 1
                self.offset = 0
                continue
            k = int(self._win_perm(self.epoch, self.rank)[self.offset])
            self.offset += 1
            self.taken += 1
            info = self.infos[si]
            nw = info["width"] // self.seq
            row = k // nw
            off = (k % nw) * self.seq
            ids, lab, seg = self._load(si)
            x = np.asarray(ids[row, off:off + self.seq], np.int64)
            y = (np.asarray(lab[row, off:off + self.seq], np.int64)
                 if lab is not None else x.copy())
            s = (np.asarray(seg[row, off:off + self.seq], np.int64)
                 if seg is not None else np.zeros(self.seq, np.int64))
            return x, y, s


def main(argv=None):
    a = parse_args(argv)
    sys.path.insert(0, REF)
    import torch
    import model_v3 as M

    torch.cuda.set_device(0)
    torch.manual_seed(int(a.init_seed))
    np.random.seed(int(a.init_seed))

    stream = WindowStream(a.data, a.seq, seed=a.seed)

    cfg_kw = dict(vocab_size=a.vocab, d_model=a.d, n_layers=a.layers, n_heads=a.heads,
                  d_ff=a.d_ff, max_seq_len=max(int(a.seq), 512), tie_embeddings=True,
                  n_experts=a.E, n_activated=a.top_k, n_shared_experts=1,
                  csa_compress=int(a.csa), csa_local_window=0, csa_conv_kernel=3,
                  moe_use_situ=True)
    cfg = M.TGAIV3Config(**cfg_kw)

    patched = 0
    if a.mode == "match":
        # 把参考实现对齐到 ORV v3_pretrain 实际建出来的那份架构
        cfg.attn_res_enabled = False
        cfg.moe_post_norm = False
        cfg.moe_balance = "sign"
        cfg.qk_clip_tau = 0.0
        cfg.qk_clip_alpha = 0.5
    model = M.TGAIV3Model(cfg).cuda()
    if a.mode == "match":
        import torch.nn as nn
        for blk in model.blocks:
            att = blk.attn
            if getattr(att, "is_global_flag", False):
                att.q_norm = nn.Identity()
                att.k_norm = nn.Identity()
                patched += 1
    model.train()
    model.enable_gradient_checkpointing(False)

    n_par = sum(p.numel() for p in model.parameters())
    kinds = [("global" if b.is_global else ("csa" if b.attn_kind == "csa" else "?"))
             for b in model.blocks]
    print("[model] mode=%s d=%d layers=%d heads=%d E=%d d_ff=%d vocab=%d tie=%s "
          "attn_res=%s moe_post_norm=%s balance=%s qk_clip_tau=%s "
          "qk_norm_global_patched_off=%d kinds=%s params=%.3fM"
          % (a.mode, a.d, a.layers, a.heads, a.E, a.d_ff, a.vocab,
             cfg.tie_embeddings, cfg.attn_res_enabled, cfg.moe_post_norm,
             cfg.moe_balance, cfg.qk_clip_tau, patched,
             "".join(k[0].upper() for k in kinds), n_par / 1e6), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=float(a.lr),
                            betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=float(a.wd))
    use_amp = (a.amp == "fp16")
    print("[optim] AdamW lr=%.3e wd=%.3g clip=%.3g amp=%s sched=constant"
          % (a.lr, a.wd, a.clip, a.amp), flush=True)

    losses, gnorms, secs, auxs = [], [], [], []
    dev = "cuda"
    for step in range(int(a.steps)):
        ids, lab, seg = stream.next_window()
        ids_t = torch.from_numpy(ids).to(dev)[None, :]
        lab_t = torch.from_numpy(lab).to(dev)[None, :]
        seg_t = torch.from_numpy(seg).to(dev)[None, :]
        t0 = time.time()
        opt.zero_grad(set_to_none=False)
        torch.cuda.synchronize()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            _logits, loss, _ = model(ids_t, targets=lab_t, seg_id=seg_t,
                                     return_qk_max=(cfg.qk_clip_tau > 0))
        aux_v = float("nan")
        if a.mode == "full" and cfg.n_experts > 1:
            aux = model.aux_loss()
            total = loss.float() + aux.float()
            aux_v = float(aux.detach().float().item())
        else:
            total = loss
        total.backward()
        if float(a.clip) > 0:
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                         float(a.clip)))
        else:
            gnorm = float("nan")
        opt.step()
        if a.mode == "full":
            import optim_v3 as O
            model.update_moe_bias()
            if cfg.qk_clip_tau > 0:
                O.apply_qk_clip(model, model._qk_max, tau=cfg.qk_clip_tau,
                                alpha=cfg.qk_clip_alpha)
        torch.cuda.synchronize()
        dt = time.time() - t0
        lv = float(loss.detach().float().item())
        losses.append(lv)
        gnorms.append(gnorm)
        auxs.append(aux_v)
        secs.append(dt)
        print("step %4d loss=%.6f gnorm=%8.3f aux=%s %.3fs"
              % (step, lv, gnorm, ("%.6f" % aux_v) if aux_v == aux_v else "-", dt),
              flush=True)

    res = dict(tag=a.tag, impl="pytorch", mode=a.mode,
               cfg=dict(d=a.d, layers=a.layers, heads=a.heads, E=a.E, d_ff=a.d_ff,
                        vocab=a.vocab, top_k=a.top_k, csa=a.csa, seq=a.seq,
                        tie=True, attn_res=cfg.attn_res_enabled,
                        moe_post_norm=cfg.moe_post_norm, balance=cfg.moe_balance,
                        qk_clip_tau=cfg.qk_clip_tau),
               params=int(n_par), lr=a.lr, steps=int(a.steps), amp=a.amp,
               clip=float(a.clip), wd=float(a.wd), seed=a.seed,
               init_seed=a.init_seed, data=a.data,
               losses=losses, gnorms=gnorms, aux=auxs,
               step_s=float(np.mean(secs)) if secs else None,
               peak_mib=int(torch.cuda.max_memory_allocated() // 1024 // 1024),
               reserved_mib=int(torch.cuda.max_memory_reserved() // 1024 // 1024))
    print("RESULT " + json.dumps({k: v for k, v in res.items()
                                  if k not in ("losses", "gnorms", "aux")},
                                 ensure_ascii=False), flush=True)
    print("losses " + json.dumps(losses), flush=True)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("[out] %s" % a.json_out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
