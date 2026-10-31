# -*- coding: utf-8 -*-
"""数据体检：训练前先看清「每条样本到底有多少有效监督」。

为什么需要它：训练侧的很多"学不动"其实是数据问题，而数据问题在 loss 曲线上看不出来。
最典型的一次：`答案 token 数 > 序列总 token 数` —— 顺着这条不可能的关系，
查出了「补位 token 被计入 loss」的 bug，它让模型大部分容量在学重复最后一个 token。

用法：
    python tools/dataset_report.py --data sft.jsonl --tokenizer tokenizer.json --seq-len 128

输出（并带若干**硬性自检**，不通过就非零退出）：
    样本数 / 总 token / 答案 token / 掩码比例 / 截断数 / 重复率 / 空答案数
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def analyze(items, encode, seq_len, key_prompt="prompt", key_completion="completion"):
    """核心统计。items 是 list[dict]；encode(str) -> list[int]。

    返回 dict；其中 `checks` 是**硬性自检**，任何一条 False 都说明数据或管线有问题。
    """
    tot, ans, trunc, empty, texts = [], [], 0, 0, []
    for it in items:
        prompt = it.get(key_prompt)
        completion = it.get(key_completion)
        if completion is None:
            text = (it.get("text") or "").strip()
            if not text:
                continue
            ids = encode(text)
            n_prompt = 0
            texts.append(text)
        else:
            p, c = str(prompt or ""), str(completion or "")
            if not c.strip():
                empty += 1
                continue
            pid, cid = encode(p), encode(c)
            ids = pid + cid
            n_prompt = len(pid)
            texts.append(p + c)
        n = len(ids)
        n_ans = n - n_prompt                      # 真正参与 loss 的 token 数
        if n > seq_len:
            trunc += 1
            n_ans = max(0, seq_len - n_prompt)
            n = seq_len
        tot.append(n)
        ans.append(n_ans)

    tot_a, ans_a = np.asarray(tot, np.int64), np.asarray(ans, np.int64)
    seen, dup = set(), 0
    for t in texts:
        if t in seen:
            dup += 1
        else:
            seen.add(t)

    checks = {
        # 这一条就是当初抓出补位 bug 的那条：不可能出现"答案 token 比总数还多"
        "答案 token ≤ 总 token": bool(np.all(ans_a <= tot_a)),
        "答案 token 中位数 > 0": bool(_pct(ans_a, 50) > 0),
        "截断比例 < 20%": bool(len(tot_a) == 0 or trunc / max(len(tot_a), 1) < 0.2),
        "重复样本 < 10%": bool(dup / max(len(texts), 1) < 0.1),
    }
    return {
        "样本数": int(len(tot_a)),
        "空答案（已跳过）": int(empty),
        "总 token 中位数": int(_pct(tot_a, 50)),
        "答案 token 中位数": int(_pct(ans_a, 50)),
        "答案 token p10/p90": "%d / %d" % (_pct(ans_a, 10), _pct(ans_a, 90)),
        "答案 token 最大": int(ans_a.max()) if len(ans_a) else 0,
        "掩码比例": "%.1f%%" % (100.0 * (1 - ans_a.sum() / max(tot_a.sum(), 1))),
        "超过 seq_len 的样本": int(trunc),
        "重复样本": int(dup),
        "checks": checks,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="只输出 JSON（便于脚本处理）")
    a = ap.parse_args(argv)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen25_sft", os.path.join(REPO, "examples", "qwen25_sft.py"))
    q = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("qwen25_sft", q)
    spec.loader.exec_module(q)
    tok = q.load_tokenizer(a.tokenizer)
    encode = lambda s: tok.encode(s).ids          # noqa: E731

    items = []
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if a.limit and len(items) >= a.limit:
                break

    rep = analyze(items, encode, a.seq_len)
    if a.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print("== 数据体检：%s（seq_len=%d）==" % (os.path.basename(a.data), a.seq_len))
        for k, v in rep.items():
            if k != "checks":
                print("  %-18s %s" % (k, v))
        print("  --- 硬性自检 ---")
        for k, ok in rep["checks"].items():
            print("  %s %s" % ("✓" if ok else "✗", k))
    bad = [k for k, ok in rep["checks"].items() if not ok]
    if bad:
        print("\n自检未过：%s\n修法：先修数据/管线再训练 —— 这类问题在 loss 曲线上看不出来。"
              % "、".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
