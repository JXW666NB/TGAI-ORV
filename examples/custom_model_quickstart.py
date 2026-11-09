# -*- coding: utf-8 -*-
"""QUICKSTART：用 ORV 训练**你自己的模型**（不需要 `.tg`，也不是 TGAI）。

这个文件就是"用户侧代码"：只依赖 `pip install tgai-orv` 暴露的公开 API。
它定义了一个很小的 Transformer（2 层、d=16），用**随机生成的合成数据**训练 20 步，
并演示本库与 torch 一致的那几件基础设施：

    AdamW + CosineLR(warmup) + 梯度裁剪 + 梯度累加

运行（CPU 也能跑，秒级）：
    python examples/custom_model_quickstart.py

与 PyTorch 的对齐证据：`tests/test_custom_model_parity.py`
—— 同一个模型（同初始化、同结构、同超参）在 ORV 与 torch 上各训练 8 步，
逐 step loss 相对误差 **1.44e-07**（判据 1e-4）。
"""
import numpy as np

import tgai_go.train as T

V, D, H, NL, SEQ = 32, 16, 4, 2, 12
BATCH = 4
STEPS = 20
LR = 1e-3
ACCUM = 2          # 梯度累加：两个微批算一次更新


class TinyTransformer(T.nn.Module):
    """最小可跑的 decoder-only Transformer（用户在 ORV 里常用写法）。"""

    def __init__(self, vocab=V, dim=D, heads=H, layers=NL):
        super().__init__()
        self.vocab, self.dim, self.heads, self.layers = vocab, dim, heads, layers
        self.emb = T.nn.Embedding(vocab, dim)
        self.ln1 = T.nn.ModuleList([T.nn.RMSNorm(dim) for _ in range(layers)])
        self.ln2 = T.nn.ModuleList([T.nn.RMSNorm(dim) for _ in range(layers)])
        self.wq = T.nn.ModuleList([T.nn.Linear(dim, dim) for _ in range(layers)])
        self.wk = T.nn.ModuleList([T.nn.Linear(dim, dim) for _ in range(layers)])
        self.wv = T.nn.ModuleList([T.nn.Linear(dim, dim) for _ in range(layers)])
        self.wo = T.nn.ModuleList([T.nn.Linear(dim, dim) for _ in range(layers)])
        self.w1 = T.nn.ModuleList([T.nn.Linear(dim, 4 * dim) for _ in range(layers)])
        self.w2 = T.nn.ModuleList([T.nn.Linear(4 * dim, dim) for _ in range(layers)])
        self.fn = T.nn.RMSNorm(dim)
        self.head = T.nn.Linear(dim, vocab)

    def forward(self, ids):
        n = len(ids)
        mask = T.tensor(np.triu(np.full((n, n), -1e9, np.float32), k=1))
        x = self.emb(T.tensor(np.asarray(ids, np.int64)))
        for i in range(self.layers):
            h = self.ln1[i](x)
            q, k, v = self.wq[i](h), self.wk[i](h), self.wv[i](h)
            sc = (q @ k.T) * (1.0 / np.sqrt(self.dim / self.heads))
            x = x + self.wo[i](T.nn.functional.softmax(sc + mask, axis=-1) @ v)
            h2 = self.ln2[i](x)
            x = x + self.w2[i](T.nn.functional.silu(self.w1[i](h2)))
        return self.head(self.fn(x))


def main():
    model = TinyTransformer()
    params = [p for p in model.parameters()]
    opt = T.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = T.optim.lr_scheduler.CosineLR(opt, T_max=STEPS, eta_min=1e-4, warmup_steps=3)
    acc = T.optim.Accumulator(params, steps=ACCUM)

    rng = np.random.default_rng(0)
    data = [rng.integers(0, V, (BATCH, SEQ + 1)) for _ in range(STEPS)]

    print("steps=%d  batch=%d  seq=%d  accum=%d  lr=%.0e" % (STEPS, BATCH, SEQ, ACCUM, LR))
    for step in range(STEPS):
        batch = data[step]
        step_loss = 0.0
        for micro in range(ACCUM):
            row = batch[micro::ACCUM]
            lg = model(row[0][:-1])
            for j in range(1, len(row)):
                lg = lg + model(row[j][:-1])
            tgt = np.concatenate([row[j][1:] for j in range(len(row))])
            loss = T.nn.functional.cross_entropy(lg * (1.0 / len(row)), tgt)
            step_loss += float(np.asarray(loss.data, np.float32))
            loss.backward()
            if not acc.tick():
                continue
            acc.finalize()                                   # 梯度按 1/ACCUM 平均
            T.nn.utils.clip_grad_norm_(params, 1.0)          # 裁剪在累加之后
            lr = sched.step()
            opt.step()
            opt.zero_grad()
            acc.reset()
            print("step %2d  loss=%.4f  lr=%.3e" % (step, step_loss / ACCUM, lr))


if __name__ == "__main__":
    main()
