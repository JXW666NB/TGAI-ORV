# -*- coding: utf-8 -*-
"""在远端 D 盘找**可用的 SFT 语料**，并打印首行片段以便**核对标记模板版本**。

为什么需要：语料文件名里的 `V2.6` / `V2.7` 不一定可信，而对话标记模板换了版本，
喂错格式会让"训练对比"变成"格式不匹配测试"。所以这里直接看内容。

用法：`python list_sft_jsonl.py [根...]`
"""
import io
import json
import os
import sys

DEFAULT_ROOTS = ["D:/tgai_nlp/data", "D:/tgai_stage2", "D:/tgai", "D:/模型库", "D:/语料"]
MIN_MB = 1.0


def sniff(path, n=260):
    """读首行，尽量取到 text 字段的开头。"""
    try:
        with io.open(path, encoding="utf-8", errors="replace") as f:
            line = f.readline().strip()
    except OSError as exc:
        return "<读取失败 %s>" % exc
    try:
        rec = json.loads(line)
        txt = rec.get("text") or rec.get("content") or ""
        if not txt:
            txt = line
    except ValueError:
        txt = line
    txt = txt.replace("\n", " ")
    return txt[:n]


def main():
    roots = sys.argv[1:] or DEFAULT_ROOTS
    rows = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d.lower() not in ("node_modules", ".git", "__pycache__",
                                                "download", "raw")]
            for fn in filenames:
                if not fn.lower().endswith((".jsonl", ".json")):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    mb = os.path.getsize(p) / 1e6
                except OSError:
                    continue
                if mb >= MIN_MB:
                    rows.append((mb, p))
    rows.sort(reverse=True)
    print("找到 %d 个 >=%.0f MB 的 jsonl/json" % (len(rows), MIN_MB))
    for mb, p in rows[:25]:
        print("\n%8.1f MB  %s" % (mb, p))
        print("          首行: %s" % sniff(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
