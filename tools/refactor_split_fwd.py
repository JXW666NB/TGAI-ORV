# -*- coding: utf-8 -*-
"""把 V27Trainer.fwd 整块替换为「fwd_attn + moe_part + fwd」拆分版（去掉已放弃的 fused 分支）。

边界识别：从 `    def fwd(self, x, i, cos, sin):` 到其后第一个同级 `    def `。
幂等：若已存在 `def fwd_attn(` 则跳过。
"""
import ast
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "python", "tgai_go", "train", "v27_train.py")

NEW = '''    def fwd_attn(self, x, i, cos, sin):
        """层的前半段：`x + attention(ln1(x))`，**不含 MoE**。

        为什么要拆（2026-11-10 实测）：MoE 的 `_combine_cuda` 有一次主机往返（D2H），
        捕获窗口内禁止 D2H，所以"整层一张图"不可能；而 attention 段没有 D2H，可以逐层
        捕获成图（28 层 attention：逐算子 966.7 ms → 图重放 140.5 ms，梯度逐位一致）。
        拆分只是把同一批算子分成两个方法，算术不变。
        """
        L = self.layers[i]
        h = L["ln1"](x)
        acc = None
        inv = 1.0 / math.sqrt(self.DH)
        for hh in range(self.NH):
            qh = T._rope_composed(L["q"][hh](h), cos, sin)
            kh = T._rope_composed(L["k"][hh](h), cos, sin)
            vh = L["v"][hh](h)
            sc = (qh @ kh.T) * inv
            pr = T.functional.softmax(sc + self._causal_mask(sc.shape[0]), axis=-1)
            part = L["o"][hh](pr @ vh)
            acc = part if acc is None else acc + part
        return x + acc

    def moe_part(self, y, i):
        """层的后半段：`y + MoE(ln2(y))`（保持逐算子，不参与图捕获）。"""
        L = self.layers[i]
        return y + L["moe"](L["ln2"](y))

    def fwd(self, x, i, cos, sin):
        return self.moe_part(self.fwd_attn(x, i, cos, sin), i)

'''


def main():
    s = io.open(TARGET, encoding="utf-8").read()
    if "def fwd_attn(" in s:
        print("已拆分（存在 fwd_attn），跳过")
        return 0
    m = re.search(r"(?m)^    def fwd\(self, x, i, cos, sin\):\n", s)
    if not m:
        print("找不到 fwd 定义")
        return 2
    start = m.start()
    nxt = re.search(r"(?m)^    def ", s[m.end():])
    if not nxt:
        print("找不到下一个方法边界")
        return 2
    end = m.end() + nxt.start()
    out = s[:start] + NEW + s[end:]
    ast.parse(out)
    io.open(TARGET, "w", encoding="utf-8", newline="\n").write(out)
    print("已替换 fwd 为拆分版（去掉 fused 分支）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
