# -*- coding: utf-8 -*-
"""Qwen2.5 在 TGAI ORV 上微调（真实开源权重，引擎原生实现）。

## 阶段

1. `--forward-only`：加载 GGUF 里的真实 Qwen2.5-0.5B 权重跑前向，与 HF 参考 logits 对齐
   （判据：相对误差 ≤1e-5）。**这是能不能微调的前提。**
2. `--steps N`：在语料上做 SFT。

## 与我们的模型（exam2）的差异

| 项 | exam2（我们的模型） | Qwen2.5 |
| --- | --- | --- |
| 注意力 | MHA | **GQA**（14 Q : 2 KV） |
| qkv | 无 bias | **有 bias** |
| 词嵌入/lm_head | 绑定 | **不绑定**（独立 output.weight） |
| RoPE | 引擎交错式（fused_rope） | 标准 rotate_half（θ=1e6） |
| MLP | MoE | 稠密 SwiGLU |

RoPE 先用**组合式** rotate_half（正确优先）；融合核版本需要把 Q/K 权重按
`tools/gguf_to_tg.maybe_permute_rope` 预置换，留作性能优化。

用法：
    python examples/qwen25_sft.py --forward-only \
        --gguf D:/DSH-Workspace/_gguf_test/qwen2.5-0.5b-fp16.gguf \
        --ref tools/_qwen_ref.npz
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)

import tgai_go.train as T  # noqa: E402
from tgai_go.train import functional as F  # noqa: E402
from tgai_go.train import nn_ops as O  # noqa: E402
from tgai_go.train import checkpoints as C  # noqa: E402

def load_converter():
    spec = importlib.util.spec_from_file_location(
        "gguf_to_tg", os.path.join(REPO, "tools", "gguf_to_tg.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gguf_to_tg"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_gguf_weights(path, cfg=None):
    """薄封装：真正的名字映射与校验在 `tgai_go.train.checkpoints`（单一事实来源）。"""
    return C.load_gguf(path).weights


# 模型与训练循环现在住在包里（单一真源，API 与示例共用同一份代码）
from tgai_go.train.qwen import (  # noqa: E402
    CFG, TINY_CFG, Qwen25, _make_xy, _probe_loss, _probe_multi, _to_np, build_sequences,
    cfg_from_checkpoint, hf_name_to_ours, load_params, load_safetensors_weights,
    load_tokenizer, make_tiny_weights, rope_inv_freq, rope_tables, save_params,
    sft, tiny_sequences)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", default="", help="HF 权重目录（含 model.safetensors）")
    ap.add_argument("--gguf", default=r"D:\DSH-Workspace\_gguf_test\qwen2.5-0.5b-fp16.gguf")
    ap.add_argument("--tg", default="", help="我们自己的 .TG 权重（训练产物/MoE 转换件）；"
                                            "与 --safetensors/--gguf 三选一，优先 --safetensors")
    ap.add_argument("--ref", default=os.path.join(REPO, "tools", "_qwen_ref.npz"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--steps", type=int, default=0, help="SFT 步数（0 表示不训）")
    ap.add_argument("--tiny", action="store_true", help="用随机权重的迷你模型验证 SFT 链路")
    ap.add_argument("--data", default="", help="jsonl 语料（user/assistant 字段）")
    ap.add_argument("--tokenizer", default="", help="tokenizer.json 路径")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--limit", type=int, default=8, help="最多取多少条样本")
    ap.add_argument("--save", default="", help="训练后保存权重的 npz 路径")
    ap.add_argument("--load", default="", help="从 npz 加载权重（生成/续训前）")
    ap.add_argument("--resume", default="",
                    help="从 npz **续训**：恢复参数 + AdamW 动量 + 步号（与 --load 的区别是后者只恢复参数）")
    ap.add_argument("--holdout", type=int, default=0,
                    help="最后 N 条留作留出集（不参与训练），探针用其中第一条")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="梯度 L2 裁剪阈值（0=不裁）。走并行归约内核，"
                         "1.36 亿元素的嵌入梯度实测 2 ms，可放心开启")
    ap.add_argument("--warmup", type=int, default=0, help="线性 warmup 步数")
    ap.add_argument("--batch", type=int, default=1, help="每步样本数（梯度累加）")
    ap.add_argument("--tiny-lr", type=float, default=3e-3)
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args(argv)

    T.set_default_device(a.device)
    print("device=%s" % (T.get_device(),))

    if a.tiny:
        cfg = TINY_CFG
        model = Qwen25(cfg, make_tiny_weights(cfg))
        seqs = tiny_sequences(cfg)
        steps = a.steps or 3
        print("tiny 模式：D=%d L=%d NH=%d KV=%d FF=%d V=%d，样本 %d 条，steps=%d"
              % (cfg["D"], cfg["L"], cfg["NH"], cfg["KV"], cfg["FF"], cfg["V"],
                 len(seqs), steps))
        probe = seqs[0]                     # 固定探针，保证可比
        losses, probes = sft(model, seqs, steps, a.tiny_lr, a.seq_len, probe=probe,
                             clip=a.clip, batch=a.batch)
        print("")
        print("训练 loss（轮转样本，不可比）：%.4f → %.4f" % (losses[0], losses[-1]))
        print("探针 loss（固定序列，可比）：  %.6f → %.6f" % (probes[0], probes[-1]))
        ok = probes[-1] < probes[0]
        print("结论：%s" % ("PASS —— SFT 链路可用（前向/反向/AdamW 全通）" if ok else "FAIL"))
        return 0 if ok else 1

    # ★ 结构参数必须**从检查点派生**，不能写死：Qwen2.5-0.5B 是 D=896，
    #   Qwen3-0.6B 是 D=1024（且 head_dim=128≠64）。写死会 reshape 崩或静默错位。
    if a.safetensors:
        print("读取 safetensors 目录：%s" % a.safetensors)
        ck = C.load_safetensors_dir(a.safetensors)
    elif a.tg:
        print("读取 .TG：%s" % a.tg)
        ck = C.load_tg(a.tg)
    else:
        print("读取 GGUF：%s" % a.gguf)
        ck = C.load_gguf(a.gguf)
    cfg = cfg_from_checkpoint(ck)
    w = dict(ck.weights)
    print("权重 %d 个；结构 L=%d D=%d NH=%d/KV=%d DH=%d FF=%d V=%d；tie=%s；QK_NORM=%s"
          % (len(w), cfg["L"], cfg["D"], cfg["NH"], cfg["KV"], cfg["DH"], cfg["FF"],
             cfg["V"], cfg["TIE"], cfg["QK_NORM"]))

    model = Qwen25(cfg, w)
    print("参数已上卡：%d 个张量" % len(model.params()))
    del w

    if a.forward_only:
        ref = np.load(a.ref)
        ids, ref_logits = ref["ids"], ref["logits"]
        ids2 = np.asarray(ids, dtype=np.int64).reshape(1, -1)
        print("参考序列长度 %d" % ids2.shape[1])
        logits = model.forward(ids2)
        got = np.asarray(logits.to_numpy(), dtype=np.float32)
        d = np.abs(got.astype(np.float64) - ref_logits.astype(np.float64))
        denom = float(np.abs(ref_logits).max()) or 1.0
        rel = float(d.max()) / denom
        print("")
        print("我们 logits[0,-1,:5] =", got[0, -1, :5])
        print("参考 logits[0,-1,:5] =", ref_logits[0, -1, :5])
        print("max|Δ| = %.3e   相对误差 = %.3e" % (float(d.max()), rel))
        ok = rel <= a.tol
        print("判据 ≤%.0e：%s" % (a.tol, "PASS" if ok else "FAIL"))
        return 0 if ok else 1

    if a.steps > 0:
        if not a.data or not a.tokenizer:
            print("SFT 需要 --data 与 --tokenizer")
            return 2
        tok = load_tokenizer(a.tokenizer)
        seqs = build_sequences(tok, a.data, a.seq_len, a.limit)
        if not seqs:
            print("语料里没有可用的 user/assistant 样本")
            return 2
        print("样本 %d 条，seq_len=%d，steps=%d，lr=%g"
              % (len(seqs), a.seq_len, a.steps, a.lr))
        if a.holdout and len(seqs) > a.holdout:
            train_seqs = seqs[:-a.holdout]
            probe = seqs[-a.holdout:]         # 整段留出集：训练时**从未见过**
            print("训练 %d 条 / 留出 %d 条（探针取自留出集，可排除'背下训练样本'）"
                  % (len(train_seqs), a.holdout))
        else:
            train_seqs = seqs
            probe = seqs[0]
        opt = T.optim.AdamW(model.params(), lr=a.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.0)
        start_step = 0
        if a.resume:
            st = load_params(model, a.resume, opt)
            start_step = int(st["step"])
            print("续训：从 step %d 继续（参数 %d 个；优化器动量%s）"
                  % (start_step, st["params"],
                     "已恢复" if st["opt"] else "未恢复（老存档只含参数）"))
        losses, probes = sft(model, train_seqs, a.steps, a.lr, a.seq_len,
                             probe=probe, clip=a.clip, warmup=a.warmup,
                             batch=a.batch, start_step=start_step, opt=opt)
        drop = (probes[0] - probes[-1]) / max(abs(probes[0]), 1e-9)
        print("")
        print("探针 loss：%.6f → %.6f（变化 %.1f%%）" % (probes[0], probes[-1], 100 * drop))
        if a.save:
            save_params(model, a.save, opt=opt, step=start_step + a.steps)
            print("权重已保存：%s" % a.save)
        gain = (probes[0] - probes[-1]) / max(abs(probes[0]), 1e-9)
        if gain >= 0.05:
            verdict = "PASS —— 留出集下降 %.1f%%，确实学到了" % (100 * gain)
            rc = 0
        elif gain > 0:
            verdict = ("未学到 —— 留出集只降 %.2f%%，在噪声范围内（判据 ≥5%%）。"
                       "修法：加步数/加语料/调 lr，别把噪声当成果" % (100 * gain))
            rc = 1
        else:
            verdict = "FAIL —— 留出集反而上升 %.1f%%（过拟合或 lr 过大）" % (-100 * gain)
            rc = 1
        print("结论：%s" % (verdict,))
        return rc

    print("没有指定动作：用 --forward-only 验证前向，或 --steps 做 SFT")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

