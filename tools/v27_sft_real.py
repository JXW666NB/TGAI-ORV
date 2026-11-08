# -*- coding: utf-8 -*-
"""真实语料 SFT 冒烟：BPE 分词（CPU） → 全参训练 N 步（GPU）。

用法（V100 上，重启后一条命令）：
    python v27_sft_real.py --steps 20 --lr 1e-6 --seq 512

与 `v27_train_run.py` 的区别：
* 语料来自**真实文本**（先用 `tgai_go.bpe.BpeTokenizer` 按 `.TG` 的 tokens/merges 编码）；
* **默认不恢复产线**（`--restore` 才恢复），避免与线上推理抢卡；
* 训练前会打印"分词回译校验"，用来判断分词是否正确 ——
  若初始 loss 远高于 `ln(V)`，第一嫌疑就是分词，而不是训练本身。

验收判据（写在 docs/TRAIN_V27_ORV.md）：
1. 初始 loss 明显低于 `ln(V) = ln(32779) ≈ 10.40`（预期 3~6）；
2. 20 步内 loss 继续下降（真实数据上不应发散）。
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)

TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7.tg"
log = RUN.log


def find_corpus(min_cjk=200, max_chars=4000):
    patterns = (r"D:\tgai_stage2\**\*.txt", r"D:\TGAI\**\*.txt", r"D:\tgai_nlp\**\*.txt")
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            try:
                if os.path.getsize(p) < 20000:
                    continue
                t = open(p, encoding="utf-8", errors="ignore").read(max_chars)
            except Exception:
                continue
            if sum("\u4e00" <= ch <= "\u9fff" for ch in t) > min_cjk:
                return p, t
    return None, ""


def encode_corpus(path, text, seq):
    from tgai_go.bpe import BpeTokenizer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(TG)
    toks, merges = info["tokens"], info["merges"]
    hdr = info.get("header") or {}
    close_tg(info)
    tk = BpeTokenizer(toks, merges, unk_id=int(hdr.get("unk_id", 0) or 0),
                      bos_id=int(hdr.get("bos_id", 1) or 1))
    ids = tk.encode(text, add_bos=True)[:seq]
    back = tk.decode(ids)
    log("语料 %s：%d 字符 → %d 个 token" % (os.path.basename(path), len(text), len(ids)))
    log("回译前 60 字：%r" % back[:60])
    same = tk.encode(back, add_bos=True)[:len(ids)] == ids
    log("分词往返一致：%s" % same)
    return np.asarray(ids, np.int64), bool(same)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--restore", action="store_true", help="跑完恢复产线（默认不恢复）")
    args = ap.parse_args()

    path, text = find_corpus()
    if not text:
        log("找不到中文语料，中止（不做假数据）")
        raise SystemExit(2)
    ids, ok = encode_corpus(path, text, args.seq)
    if len(ids) < 32:
        log("分词结果过短（%d），中止" % len(ids))
        raise SystemExit(2)
    if not ok:
        log("警告：分词往返不一致，loss 数值可能不可信")

    if not RUN.isolate():
        log("GPU 隔离失败，中止")
        if args.restore:
            RUN.restore()
        raise SystemExit(3)
    try:
        import torch

        import tgai_go.train as T
        from tgai_go.train.opt_fp16mv import Fp16MomAdamW
        from tgai_go.train.v27_train import V27Trainer

        T.set_default_device("cuda")
        ck = torch.load(RUN.PT, map_location="cpu", mmap=True, weights_only=False)
        mc, sd = ck["model_config"], ck["model_state_dict"]
        cfg = dict(d_model=int(mc["d_model"]), n_heads=int(mc["n_heads"]), d_ff=int(mc["d_ff"]),
                   n_experts=int(mc["n_experts"]), n_activated=int(mc["n_activated"]),
                   n_layers=int(mc["n_layers"]), rope_theta=float(mc.get("rope_theta", 1e4)))
        NL = cfg["n_layers"]
        tr = V27Trainer(cfg, n_layers=NL)
        tr.load_all(sd)
        tr.attach_head(sd)
        log("装载 28 层 + head：%d MiB" % RUN.vram())
        opts = [Fp16MomAdamW(tr.params_of(i), lr=args.lr) for i in range(NL)]
        head_opt = Fp16MomAdamW(tr.head_params(), lr=args.lr)
        log("优化器就绪：%d MiB" % RUN.vram())

        D, DH = cfg["d_model"], cfg["d_model"] // cfg["n_heads"]
        TK = len(ids) - 1
        j = np.arange(0, DH, 2, dtype=np.float32) / DH
        ang = np.repeat(np.arange(TK, dtype=np.float32)[:, None]
                        * (cfg["rope_theta"] ** (-j))[None, :], 2, axis=1)
        cos = T.tensor(np.cos(ang).astype(np.float32))
        sin = T.tensor(np.sin(ang).astype(np.float32))
        emb = sd["token_embedding.weight"].float().numpy()
        x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)
        targets = ids[1:]

        losses = []
        t0 = time.time()
        for step in range(args.steps):
            xs = tr.forward_all(x0, cos, sin)
            loss, g = tr.loss_and_head_backward(xs[-1], targets, cos, sin)
            lv = float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss))
            losses.append(lv)
            head_opt.step()
            head_opt.zero_grad()
            for p in tr.head_params():
                p.grad = None
            tr.backward_and_step(xs, g, cos, sin, opts)
            if step % 2 == 0 or step == args.steps - 1:
                log("step %2d  loss=%.4f  显存 %d MiB  %.1fs"
                    % (step, lv, RUN.vram(), time.time() - t0))
            if step % 10 == 9:
                T._cuda().empty_cache()
        el = time.time() - t0
        lnv = float(np.log(len(sd["token_embedding.weight"])))
        log("真实语料 %d 步：%.4f → %.4f；ln(V)=%.2f" % (args.steps, losses[0], losses[-1], lnv))
        log("判定：初始 loss %s ln(V) ⇒ 分词%s"
            % ("低于" if losses[0] < lnv - 1 else "接近/高于",
               "合理" if losses[0] < lnv - 1 else "很可能有问题"))
    finally:
        try:
            tr = None
            opts = None
            head_opt = None
            T._cuda().empty_cache()
        except Exception:
            pass
        if args.restore:
            for _ in range(12):
                time.sleep(5)
                if RUN.vram() < 2000:
                    break
            RUN.restore()


if __name__ == "__main__":
    main()
