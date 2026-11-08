# -*- coding: utf-8 -*-
"""真实语料**评估**（不训练、不更新参数）：衡量 V2.7 在留出 SFT/预训练样本上的 CE loss。

为什么单独一个脚本：
* 用户要求"正式训练前先确认质量"，所以先用**只读评估**给出客观数字；
* 评估不需要优化器（省 11 GB），只需权重 11.3 GB，显存更宽松；
* 与 `v27_sft_real.py` 共用 BPE 与装载路径，保证口径一致。

用法：
    python v27_eval_real.py --samples 8 --seq 512

输出：每条样本的 loss/token 数，以及均值、最好、最差；并给出 `ln(V)=10.40` 参照。
"""
import argparse
import glob
import json
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


def load_tokenizer():
    from tgai_go.bpe import BOS_ID, UNK_ID, BpeTokenizer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(TG)
    tk = BpeTokenizer(info["tokens"], info["merges"], unk_id=UNK_ID, bos_id=BOS_ID)
    close_tg(info)
    return tk


def collect_samples(tk, want, seq, skip=20):
    """从 SFT 语料库取样本；`skip` 跳过前面的记录，尽量避开训练时用过的那几条。"""
    out = []
    for p in sorted(glob.glob(os.path.join("D:\\SFT\u8bed\u6599\u5e93", "**", "*.jsonl"), recursive=True)):
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                idx = 0
                for line in f:
                    if not line:
                        break
                    if idx < skip:
                        idx += 1
                        continue
                    idx += 1
                    try:
                        t = (json.loads(line).get("text") or "")
                    except Exception:
                        continue
                    if len(t) < 120:
                        continue
                    ids = tk.encode(t, add_bos=True)[:seq]
                    if len(ids) >= 64:
                        out.append((os.path.basename(p), ids))
                    if len(out) >= want:
                        return out
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--skip", type=int, default=20)
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    tk = load_tokenizer()
    samples = collect_samples(tk, args.samples, args.seq, args.skip)
    if not samples:
        log("取不到样本，中止")
        raise SystemExit(2)
    log("取到 %d 条留出样本（seq<=%d）" % (len(samples), args.seq))

    if not RUN.isolate():
        log("GPU 隔离失败，中止")
        if args.restore:
            RUN.restore()
        raise SystemExit(3)
    try:
        import torch

        import tgai_go.train as T
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

        D, DH = cfg["d_model"], cfg["d_model"] // cfg["n_heads"]
        emb = sd["token_embedding.weight"].float().numpy()
        losses = []
        for name, ids in samples:
            ids = np.asarray(ids, np.int64)
            TK = len(ids) - 1
            if TK < 8:
                continue
            j = np.arange(0, DH, 2, dtype=np.float32) / DH
            ang = np.repeat(np.arange(TK, dtype=np.float32)[:, None]
                            * (cfg["rope_theta"] ** (-j))[None, :], 2, axis=1)
            cos = T.tensor(np.cos(ang).astype(np.float32))
            sin = T.tensor(np.sin(ang).astype(np.float32))
            x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)
            xs = tr.forward_all(x0, cos, sin)
            loss, _ = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
            lv = float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss))
            losses.append(lv)
            log("  %-34s %3d tok  loss=%.4f" % (name[:34], TK, lv))
            try:
                T._cuda().empty_cache()
            except Exception:
                pass
        lnv = float(np.log(len(sd["token_embedding.weight"])))
        log("评估结果：均值 %.4f，最好 %.4f，最差 %.4f（ln(V)=%.2f）"
            % (float(np.mean(losses)), float(np.min(losses)), float(np.max(losses)), lnv))
        log("相对均匀分布：好 %.2f 倍（越低越好）" % (lnv / float(np.mean(losses))))
    finally:
        try:
            tr = None
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
