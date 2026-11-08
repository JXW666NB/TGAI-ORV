# -*- coding: utf-8 -*-
"""TGAI 的 BPE 分词器（纯 Python，**不依赖引擎/DLL**）。

为什么需要：训练时要把真实语料变成 token id，而 `tgai_go.load()` 那条路要求引擎把
3B 权重装进显存（13 GB），与训练不能同时进行；而且实测在独立进程里 `encode()` 返回 0 个 id
（引擎未真正加载模型）。所以这里按 `.TG` 里自带的两张表（tokens / merges）**复刻同样的 BPE**。

算法（sentencepiece 风格，与 V2.7 词表的 `▁` 约定一致）：
1. 规范化：空格 → `▁`(U+2581)；文本开头补一个 `▁`
2. 按字符切成初始片段；词表里没有的字符 → 走 `<unk>`
3. 反复合并"**合并优先级最高**（rank 最小）"的相邻对，直到没有可合并对
4. 片段 → id（用 token→id 映射）

`merges` 的两种常见形态都接受：`"a b"`（空格分隔）或 `(a, b)` 元组。
"""
from __future__ import annotations

SPACE = "\u2581"          # ▁


def _parse_merge(m):
    if isinstance(m, (tuple, list)):
        if len(m) != 2:
            raise ValueError("merge 必须是二元组，收到 %r。修法：检查 .TG 的 merges 段" % (m,))
        return str(m[0]), str(m[1])
    s = str(m)
    if " " in s:
        a, b = s.split(" ", 1)
        return a, b
    if "\t" in s:
        a, b = s.split("\t", 1)
        return a, b
    raise ValueError("无法解析的 merge：%r。修法：确认 merges 是 'a b' 或 (a,b)" % (m,))


class BpeTokenizer(object):
    def __init__(self, tokens, merges, unk_id=None, bos_id=None, eos_id=None):
        self.tokens = list(tokens)
        self.tok2id = {}
        for i, t in enumerate(self.tokens):
            self.tok2id.setdefault(t, i)
        self.ranks = {}
        for r, m in enumerate(merges):
            a, b = _parse_merge(m)
            self.ranks.setdefault((a, b), r)
        self.unk_id = int(unk_id) if unk_id is not None else 0
        self.bos_id = bos_id
        self.eos_id = eos_id

    # ---------- 编码 ----------
    def _normalize(self, text):
        return text.replace(" ", SPACE)

    def _merge_pieces(self, pieces):
        """按 rank 从小到大反复合并相邻对（每次取全句最优的一对）。"""
        while len(pieces) > 1:
            best, best_rank = None, None
            for i in range(len(pieces) - 1):
                key = (pieces[i], pieces[i + 1])
                r = self.ranks.get(key)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = key, r
            if best is None:
                break
            # 合并**所有**出现位置（与 sentencepiece 一致）
            out, i = [], 0
            while i < len(pieces):
                if i + 1 < len(pieces) and pieces[i] == best[0] and pieces[i + 1] == best[1]:
                    out.append(best[0] + best[1])
                    i += 2
                else:
                    out.append(pieces[i])
                    i += 1
            pieces = out
        return pieces

    def encode(self, text, add_bos=False):
        if not isinstance(text, str):
            raise TypeError("encode 只接受 str，收到 %r。修法：先 decode 成文本" % type(text))
        s = SPACE + self._normalize(text)
        pieces = list(s)
        pieces = self._merge_pieces(pieces)
        ids = [self.tok2id.get(p, self.unk_id) for p in pieces]
        if add_bos and self.bos_id is not None:
            ids = [int(self.bos_id)] + ids
        return ids

    # ---------- 解码 ----------
    def decode(self, ids):
        parts = []
        for i in ids:
            i = int(i)
            if 0 <= i < len(self.tokens):
                parts.append(self.tokens[i])
        return "".join(parts).replace(SPACE, " ").strip()

    def roundtrip_ok(self, ids):
        """编码→解码→再编码，id 序列是否一致（判断分词是否可逆的自检）。"""
        return self.encode(self.decode(ids), add_bos=False) == list(ids)
