# -*- coding: utf-8 -*-
"""把 CoT 语料切成固定长度的 token 窗口，供 ORV / PyTorch **两侧共用同一份 id**。

## 关键约定（决定对比是否公平）

* 只做**一次**分词，结果落盘成 `.npy`；两侧都读同一份文件，**不允许各自 tokenize**；
* 语料 `text` 里已经带好对话模板（`用户:` / `TGAI?【思考过程】`），所以不再套模板；
* 每条样本尾部补 `eos`，样本之间互不重叠地切成 `seq` 长窗口；
* 标签 = 输入（**全序列语言模型损失**）。两侧相同，所以曲线可比；
  注意这**不等于** GO 的 answer-only SFT 口径，文档里会标注。

只读 + CPU，不需要显卡。

用法：`python sft_prep.py <bench_cfg.json>`
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    cfg_path = sys.argv[1]
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    from tgai_go.read_tg import read_tg, close_tg
    from tgai_go.bpe import BpeTokenizer

    info = read_tg(cfg["model_tg"])
    try:
        tok = BpeTokenizer(info["tokens"], info["merges"])
        vocab = len(info["tokens"])
        eos = int(info["header"].get("eos") or 2)
        pad = int(info["header"].get("pad") or 0)
    finally:
        close_tg(info)
    print("分词器: vocab=%d eos=%d pad=%d" % (vocab, eos, pad))

    stream = []
    samples = 0
    with open(cfg["corpus"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            text = rec.get("text") or ""
            if not text:
                continue
            stream.extend(tok.encode(text))
            stream.append(eos)
            samples += 1
            if samples >= 4000:          # 只需要几十个窗口，4000 条足够
                break
    print("样本数: %d, token 总数: %d" % (samples, len(stream)))

    os.makedirs(cfg["outdir"], exist_ok=True)
    meta = {"samples": samples, "tokens": len(stream), "vocab": vocab, "eos": eos}
    for seq in cfg["seqs"]:
        n_win = min(48, len(stream) // seq)
        if n_win <= 0:
            print("  seq=%d 窗口不足（token 只有 %d）" % (seq, len(stream)))
            continue
        arr = np.empty((n_win, seq), dtype=np.int32)
        for i in range(n_win):
            arr[i] = stream[i * seq:(i + 1) * seq]
        path = os.path.join(cfg["outdir"], "ids_%d.npy" % seq)
        np.save(path, arr)
        print("  seq=%-5d 窗口=%d → %s（%.1f MB）"
              % (seq, n_win, path, arr.nbytes / 1e6))
        meta["win_%d" % seq] = n_win

    with open(os.path.join(cfg["outdir"], "prep_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("完成 →", cfg["outdir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
