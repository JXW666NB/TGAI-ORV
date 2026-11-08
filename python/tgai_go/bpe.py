# -*- coding: utf-8 -*-
"""TGAI 的 BPE 分词器（纯 Python，**不依赖引擎/DLL**）。

`.TG` 里的两张表（实测，V2.7）：
* `tokens`：32779 个字符串，前 6 个是 `<PAD> <UNK> <BOS> <EOS> <IMG> <AUD>`
* `merges`：24355 条，形式是**整数 id 对** `(93, 81)`，不是字符串！

这对应 sentencepiece 的典型约定：
```
id:  [0..5]  特殊符号
     [6 .. base-1]        基础符号（字符/字节）
     [base .. V-1]        合并产物，其中 merges[i] 合并出的 id = base + i
base = len(tokens) - len(merges)      # V2.7: 32779 - 24355 = 8424
```
所以编码必须先按**字符**映射成 id，再按 rank 从小到大合并 **id 对**，
合并结果 id = `base + rank`；解码则直接 `tokens[id]`。

早期版本按字符串合并 → 合并规则全部失效、大量 `<UNK>`、回译不一致（已实测到）。
"""
from __future__ import annotations

import struct

SPACE = "\u2581"          # ▁
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def _as_id_pair(m):
    """把一条 merge 归一化成 (a:int, b:int)。字符串形式（'a b'）也接受，用于兼容别的词表。"""
    if isinstance(m, (tuple, list)) and len(m) == 2:
        a, b = m
        if isinstance(a, int) and isinstance(b, int):
            return a, b
        return str(a), str(b)
    if isinstance(m, str):
        for sep in (" ", "\t"):
            if sep in m:
                a, b = m.split(sep, 1)
                try:
                    return int(a), int(b)
                except ValueError:
                    return a, b
        raise ValueError("无法解析的 merge：%r。修法：确认 merges 是 id 对或 'a b'" % (m,))
    raise ValueError("无法解析的 merge：%r" % (m,))


class BpeTokenizer(object):
    def __init__(self, tokens, merges, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID):
        self.tokens = list(tokens)
        self.unk_id = int(unk_id)
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.tok2id = {}
        for i, t in enumerate(self.tokens):
            self.tok2id.setdefault(t, i)

        pairs = [_as_id_pair(m) for m in merges]
        self.id_mode = bool(pairs) and all(isinstance(a, int) for a, _ in pairs)
        if self.id_mode:
            self.base = len(self.tokens) - len(pairs)
            if self.base <= 0:
                raise ValueError("merges 数 (%d) 不少于 tokens 数 (%d)，词表结构不符"
                                 % (len(pairs), len(self.tokens)))
            self.ranks = {}
            # 合并产物的 id **不能**假定是 base+rank：实测 V2.7 上 TGAI 会解成别的词。
            # 稳妥做法：用两个子符号的字符串拼起来反查真实 id，查不到才退回 base+rank。
            self.result = {}
            miss = 0
            for r, (a, b) in enumerate(pairs):
                self.ranks.setdefault((a, b), r)
                sa = self.tokens[a] if 0 <= a < len(self.tokens) else None
                sb = self.tokens[b] if 0 <= b < len(self.tokens) else None
                rid = None
                if sa is not None and sb is not None:
                    rid = self.tok2id.get(sa + sb)
                if rid is None or rid < self.base:
                    rid = self.base + r
                    miss += 1
                self.result.setdefault(r, rid)
            self.result_misses = miss
        else:                                   # 字符串模式（兼容其它词表）
            self.base = len(self.tokens)
            self.ranks = {}
            for r, (a, b) in enumerate(pairs):
                self.ranks.setdefault((self.tok2id.get(a, self.unk_id),
                                       self.tok2id.get(b, self.unk_id)), r)

    # ---------- 编码 ----------
    @staticmethod
    def _normalize(text):
        return text.replace(" ", SPACE)

    def _initial_ids(self, s):
        ids = []
        for ch in s:
            i = self.tok2id.get(ch)
            ids.append(self.unk_id if i is None or i >= self.base else i)
        return ids

    def _merge_ids(self, ids):
        """按 rank 从小到大反复合并相邻 id 对（每次取全句最优，合并所有出现位置）。"""
        while len(ids) > 1:
            best, best_rank = None, None
            for i in range(len(ids) - 1):
                key = (ids[i], ids[i + 1])
                r = self.ranks.get(key)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = key, r
            if best is None:
                break
            new_id = self.result[best_rank]
            out, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == best[0] and ids[i + 1] == best[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def encode(self, text, add_bos=False):
        if not isinstance(text, str):
            raise TypeError("encode 只接受 str，收到 %r。修法：先 decode 成文本" % type(text))
        s = SPACE + self._normalize(text)
        ids = self._merge_ids(self._initial_ids(s))
        if add_bos and self.bos_id is not None:
            ids = [int(self.bos_id)] + ids
        return ids

    # ---------- 解码 ----------
    def decode(self, ids, strip=True):
        parts = []
        for i in ids:
            i = int(i)
            if 0 <= i < len(self.tokens):
                parts.append(self.tokens[i])
        s = "".join(parts).replace(SPACE, " ")
        return s.strip() if strip else s

    def roundtrip_ok(self, ids):
        """编码→解码→再编码，id 序列是否一致（去掉 BOS 后比较）。"""
        core = [int(i) for i in ids if int(i) != (self.bos_id if self.bos_id is not None else -1)]
        again = self.encode(self.decode(core), add_bos=False)
        return again == core
