# -*- coding: utf-8 -*-
"""新手四步：导入语料 → 训练分词表 → 建模型 → 预训练。

这份脚本**不需要 GPU**，也不需要先准备语料 —— 第一次运行会自动生成一份示例语料，
让你先看清楚每一步在做什么，再换成自己的数据。

    python examples/beginner_pipeline.py                 # 用自带的示例语料
    python examples/beginner_pipeline.py my_corpus.txt   # 用自己的语料

每一步都对应 README 里的一句话，顺序不要颠倒：
  ① 语料 -> ② 分词表 -> ③ token id -> ④ 训练
"""
import os
import sys

import numpy as np

import tgai_go.train as T

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo_corpus.txt")


def make_demo_corpus(path):
    """造一份**有规律**的示例语料。

    为什么要"有规律"：纯随机文本的 loss 只能降到 ln(vocab) 附近，看曲线会误以为没学会。
    这里用固定词表随机拼句，既有变化又能被模型学到，loss 会明显下降。
    """
    rng = np.random.default_rng(0)
    subjects = ["模型", "数据", "梯度", "注意力", "参数", "样本"]
    verbs = ["优化", "训练", "收敛", "更新", "缩放", "归一"]
    objects = ["损失", "权重", "分布", "特征", "效率", "精度"]
    lines = []
    # 只用 600 句：这个纯 Python BPE 每合并一次就要扫一遍全文本，
    # 语料越大越慢（600 句约几秒，3000 句要几分钟）。示例够用就行。
    for _ in range(600):
        lines.append("%s%s了%s。" % (rng.choice(subjects), rng.choice(verbs), rng.choice(objects)))
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return path


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        path = make_demo_corpus(DEMO)
        print("[0] 没给语料，已生成示例：%s" % path)

    # ① 导入语料（纯文本或 .jsonl 都行）
    text = T.load_corpus(path)
    print("[1] 语料 %d 字符" % len(text))

    # ② 在你自己的语料上训练分词表（不需要外部 tokenizer）
    tok = T.train_tokenizer(text, vocab_size=400)
    print("[2] 分词表：%r" % (tok,))

    # ③ 语料 -> token id -> 定长切块
    ids = tok.encode(text)
    data = T.tokens(ids, seq=64)
    print("[3] token 数 %d，切出 %d 个训练块" % (len(ids), len(data)))

    # ④ 建模型 + 预训练（这几个数字就是你要调的超参）
    T.set_default_device("cpu")
    model = T.preset.transformer(vocab=tok.vocab_size, d_model=64, n_layers=2, n_heads=4)
    print("[4] 模型参数量 %d，开始训练……" % model.n_params())
    hist = T.fit(model, data, steps=80, lr=5e-3, log_every=20)

    first, last = hist["loss"][0], hist["loss"][-1]
    print("\nloss: %.4f -> %.4f（降到原来的 %.0f%%）" % (first, last, 100.0 * last / max(first, 1e-9)))
    if last >= first * 0.9:
        print("⚠️  loss 几乎没降，通常说明 lr 太小 / 数据没规律 / steps 太少")
    else:
        print("✔ 训练链路是通的。把这个脚本换成你的语料和超参就能开始真正的预训练。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
