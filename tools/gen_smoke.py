# -*- coding: utf-8 -*-
"""端到端生成冒烟：加载 .TG → 编码 → 生成 → 解码。

## 为什么需要

`probe_infer.py` 只证明"能加载"。但"推理可用"的判据是**真的出字** ——
外部报告里正是因为加载就失败，压根走不到这一步。

这个脚本对每个给定 `.TG`：
1. 打印头版本（3/4）；
2. 加载并打印模型信息；
3. 跑一次短生成，打印原文与解码结果，并断言**输出非空且不是全零**。

用法：
    python gen_smoke.py <model.tg> ["提示词"] [--max-tokens 16]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))


def head_version(path):
    import struct
    with open(path, "rb") as fh:
        head = fh.read(8)
    magic, ver = struct.unpack("<II", head)
    return magic, ver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompt", nargs="?", default="你好")
    ap.add_argument("--max-tokens", type=int, default=16)
    a = ap.parse_args()

    import tgai_go
    magic, ver = head_version(a.model)
    print("文件: %s" % a.model)
    print("  头: magic=%#x version=%d" % (magic, ver))

    with tgai_go.load(a.model) as m:
        print("  模型: layers=%s d_model=%s vocab=%s max_seq=%s"
              % (m.n_layers, m.d_model, m.vocab_size, m.max_seq_len))
        ids = m.encode(a.prompt)
        print("  encode(%r) -> %d ids: %s..." % (a.prompt, len(ids), list(ids)[:8]))
        out = m.generate(a.prompt, max_tokens=a.max_tokens)
        text = out if isinstance(out, str) else str(out)
        print("  generate -> %r" % (text[:120],))
        ok = bool(text.strip())
        print("  判读: %s" % ("出字了，推理可用" if ok else "**输出为空**"))
        return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
