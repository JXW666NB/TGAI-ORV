# -*- coding: utf-8 -*-
"""next-token 预测检查：用**真实分词 + ORV 训练栈的前向**，看模型预测是否合理。

这是"质量"的直接证据（比单独看 loss 更有说服力）：
* 若整条链路（权重装载 / RoPE / MoE 路由 / BPE 分词）有问题，预测会明显跑偏；
* 不需要生成循环、不需要优化器，只一次前向（11.3 GB 显存）。

用法：
    python v27_next_token.py            # 用内置几条中文前缀
    python v27_next_token.py --topk 8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)

TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7.tg"
log = RUN.log

PROMPTS = [
    "\u7528\u6237:\u4e2d\u56fd\u7684\u9996\u90fd\u662f\u54ea\u91cc\uff1f\nTGAI?",
    "\u7528\u6237:1+1\u7b49\u4e8e\u51e0\uff1f\nTGAI?",
    "\u7528\u6237:\u4f60\u53eb\u4ec0\u4e48\uff1f\nTGAI?",
    "\u7528\u6237:\u7528\u4e00\u53e5\u8bdd\u89e3\u91ca\u4ec0\u4e48\u662f\u8fc7\u62df\u5408\u3002\nTGAI?",
]


def corpus_check(tr, tk, sd, cfg, D, DH_to_use=None, n_samples=3, seq=512):
    """真实语料 teacher forcing：整段前向一次，算 mean CE + top1/top5 命中率。"""
    import glob, json
    import tgai_go.train as T
    DH = D // cfg["n_heads"]
    emb = sd["token_embedding.weight"].float().numpy()
    got = []
    for p in sorted(glob.glob(os.path.join("D:\\SFT\u8bed\u6599\u5e93", "**", "*.jsonl"), recursive=True)):
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line:
                        break
                    try:
                        t = (json.loads(line).get("text") or "")
                    except Exception:
                        continue
                    if len(t) < 200:
                        continue
                    ids = tk.encode(t, add_bos=True)[:seq]
                    if len(ids) >= 128:
                        got.append(ids)
                    if len(got) >= n_samples:
                        break
        except Exception:
            continue
        if len(got) >= n_samples:
            break
    for ids in got:
        ids = np.asarray(ids, np.int64)
        TK = len(ids) - 1
        j = np.arange(0, DH, 2, dtype=np.float32) / DH
        ang = np.repeat(np.arange(TK, dtype=np.float32)[:, None]
                        * (cfg["rope_theta"] ** (-j))[None, :], 2, axis=1)
        cos = T.tensor(np.cos(ang).astype(np.float32))
        sin = T.tensor(np.sin(ang).astype(np.float32))
        xs = tr.forward_all(np.ascontiguousarray(emb[ids[:-1]], np.float32), cos, sin)
        logits = tr.head(tr.final_norm(T.tensor(xs[-1])))
        lg = np.asarray(logits.to_numpy() if hasattr(logits, "to_numpy") else logits, np.float32)
        tgt = ids[1:]
        m = lg.max(axis=-1, keepdims=True)
        e = np.exp(lg - m)
        pr = e / e.sum(axis=-1, keepdims=True)
        nll = -np.log(np.maximum(pr[np.arange(len(tgt)), tgt], 1e-9)).mean()
        top1 = float((lg.argmax(-1) == tgt).mean())
        top5 = float(np.mean([tgt[i] in np.argsort(-lg[i])[:5] for i in range(len(tgt))]))
        RUN.log("  样本 %3d tok  CE=%.4f  top1=%.1f%%  top5=%.1f%%" % (TK, nll, top1*100, top5*100))
        del logits, lg, pr, e
        try:
            T._cuda().empty_cache()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--corpus", action="store_true", help="用真实语料做 teacher forcing")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    from tgai_go.bpe import BOS_ID, UNK_ID, BpeTokenizer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(TG)
    tk = BpeTokenizer(info["tokens"], info["merges"], unk_id=UNK_ID, bos_id=BOS_ID)
    close_tg(info)
    log("分词器就绪：base=%d merges=%d" % (tk.base, len(tk.ranks)))

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
        if args.corpus:
            corpus_check(tr, tk, sd, cfg, D)
            return
        emb = sd["token_embedding.weight"].float().numpy()
        for prompt in PROMPTS:
            ids = np.asarray(tk.encode(prompt, add_bos=True)[-220:], np.int64)
            TK = len(ids)
            j = np.arange(0, DH, 2, dtype=np.float32) / DH
            ang = np.repeat(np.arange(TK, dtype=np.float32)[:, None]
                            * (cfg["rope_theta"] ** (-j))[None, :], 2, axis=1)
            cos = T.tensor(np.cos(ang).astype(np.float32))
            sin = T.tensor(np.sin(ang).astype(np.float32))
            x0 = np.ascontiguousarray(emb[ids], np.float32)
            xs = tr.forward_all(x0, cos, sin)
            h = T.tensor(xs[-1][-1:], requires_grad=False)
            logits = tr.head(tr.final_norm(h))
            lg = np.asarray(logits.to_numpy() if hasattr(logits, "to_numpy") else logits,
                            np.float32)[0]
            p = np.exp(lg - lg.max())
            p /= p.sum()
            top = np.argsort(-p)[: args.topk]
            shown = ["%r(%.3f)" % (tk.decode([int(t)]), float(p[t])) for t in top]
            log("前缀 %r" % prompt.replace("\n", "\\n"))
            log("   预测 top%d: %s" % (args.topk, " ".join(shown)))
            del h, logits
            try:
                T._cuda().empty_cache()
            except Exception:
                pass
    finally:
        try:
            tr = None
            T._cuda().empty_cache()
        except Exception:
            pass
        if args.restore:
            import time
            for _ in range(12):
                time.sleep(5)
                if RUN.vram() < 2000:
                    break
            RUN.restore()


if __name__ == "__main__":
    main()
