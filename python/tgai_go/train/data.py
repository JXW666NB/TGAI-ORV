# -*- coding: utf-8 -*-
"""数据层：把"别人的数据"变成我们训练循环吃的形状。

为什么需要单独一层：训练脚本里现在直接读 jsonl、按 `{"prompt","completion"}` 解析，
别人手上的数据可能是：
  * HuggingFace `datasets.Dataset` / `IterableDataset`；
  * `{"text": ...}` 纯文本（预训练语料）；
  * `{"messages": [{"role","content"}, ...]}` 对话格式（ShareGPT/OpenAI 导出）；
  * 超大的 jsonl（不能一次读进内存）。

这一层只做**规范化与流式读取**，不碰分词、不碰模型 —— 这样它能在 CPU 上被完整测试，
也不会把训练脚本和数据格式焊死在一起。
"""
from __future__ import annotations

import io
import json
import os


class DataError(ValueError):
    """数据格式问题。消息里必须写清「哪一行 / 什么形状 / 怎么修」。"""


# ---------------------------------------------------------------- 流式读 jsonl

def read_jsonl(path, limit=None):
    """逐行读 jsonl（**不会**一次全读进内存）。坏行会指出行号与修法。"""
    if not os.path.isfile(path):
        raise DataError("数据文件不存在：%s。修法：检查路径，或用 tools/dataset_report.py 先体检"
                        % (path,))
    out = []
    with io.open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise DataError(
                    "第 %d 行不是合法 JSON（%s）：%s…。修法：确认没有多余逗号/未转义引号；"
                    "大文件可用 jq 逐行校验" % (lineno, e.msg, line[:60])) from None
            if not isinstance(obj, dict):
                raise DataError("第 %d 行应是 JSON 对象，实际是 %s。修法：每行一个 {…}"
                                % (lineno, type(obj).__name__))
            out.append(obj)
            if limit and len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------- 规范化

def _flatten_messages(msgs):
    """把 [{"role","content"}, ...] 拼成 prompt/completion。

    约定：**最后一条 assistant** 作为答案，它之前的全部作为提示。
    """
    if not isinstance(msgs, list) or not msgs:
        raise DataError("messages 必须是非空列表。修法：每条形如 "
                        "[{\"role\":\"user\",\"content\":\"…\"},"
                        " {\"role\":\"assistant\",\"content\":\"…\"}]")
    last_asst = None
    for i in range(len(msgs) - 1, -1, -1):
        if str(msgs[i].get("role", "")).lower() == "assistant":
            last_asst = i
            break
    if last_asst is None:
        raise DataError("messages 里没有 assistant 轮（无法确定要学什么）。"
                        "修法：至少放一条 role=assistant 的回复")

    def txt(m):
        c = m.get("content")
        if isinstance(c, list):                       # 多段内容（OpenAI 风格）
            return "".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
        return str(c or "")

    prompt = "".join(txt(m) + "\n" for m in msgs[:last_asst])
    return prompt.rstrip("\n"), txt(msgs[last_asst])


def normalize(items, prompt_key="prompt", completion_key="completion",
              text_key="text"):
    """把各种形态统一成 `[{"prompt":…, "completion":…}]`。

    支持：
      * `{"prompt","completion"}`（原生）
      * `{"text"}`                     —— 无监督：整段都是 loss（prompt 置空）
      * `{"messages":[…]}`             —— 对话格式，最后一条 assistant 为答案
      * `{"instruction","output"}`     —— Alpaca 风格
    认不出来会**明确报错**并列出支持的键，而不是静默丢弃。
    """
    out = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise DataError("第 %d 条不是 dict，是 %s。修法：见 normalize 的文档"
                            % (i, type(it).__name__))
        # 注意：要区分「有这个键但值为空」和「根本没有这个键」——
        # 前者是数据问题（空答案），后者是格式问题。测试逼出来的区分。
        if completion_key in it:
            comp = str(it.get(completion_key) or "")
            if not comp.strip():
                raise DataError(
                    "第 %d 条的 %s 为空。修法：过滤掉空答案样本 —— "
                    "空答案会让 loss mask 全关，那一批等于没学"
                    % (i, completion_key))
            out.append({"prompt": str(it.get(prompt_key) or ""), "completion": comp})
        elif text_key in it and str(it.get(text_key) or "").strip():
            out.append({"prompt": "", "completion": str(it[text_key])})
        elif isinstance(it.get("messages"), list):
            p, c = _flatten_messages(it["messages"])
            if not c.strip():
                raise DataError("第 %d 条最后一条 assistant 的内容为空。修法：过滤该样本" % (i,))
            out.append({"prompt": p, "completion": c})
        elif it.get("instruction") not in (None, ""):
            comp = str(it.get("output") or it.get("response") or "")
            if not comp.strip():
                raise DataError(
                    "第 %d 条有 instruction 但没有 output/response。修法：补上答案，或过滤该样本"
                    % (i,))
            out.append({"prompt": str(it["instruction"]), "completion": comp})
        else:
            raise DataError(
                "第 %d 条认不出格式（键：%s）。修法：改写成 {\"prompt\",\"completion\"}，"
                "或 {\"text\"}，或 {\"messages\":[…]}，或 {\"instruction\",\"output\"}"
                % (i, sorted(it.keys())[:8]))
    return out


# ---------------------------------------------------------------- HF datasets

def from_hf(dataset, prompt_key="prompt", completion_key="completion",
            text_key="text", limit=None):
    """把 HuggingFace 的 `datasets.Dataset` / `IterableDataset` 转成规范化样本。

    这里用**鸭子类型**（有 `__len__`/`__getitem__` 或可迭代即可），
    所以不 import `datasets`，CI 里没装也能测（用假的 dataset 对象）。
    """
    rows = []
    try:
        it = iter(dataset)
    except TypeError:
        raise DataError("传入的对象不可迭代（%s）。修法：传 datasets.Dataset、list 或生成器"
                        % type(dataset).__name__) from None
    for row in it:
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return normalize(rows, prompt_key, completion_key, text_key)


# ---------------------------------------------------------------- 划分

def train_eval_split(items, holdout=8, seed=0):
    """确定性地切出留出集（同 seed 可复现），返回 (train, eval)。

    与训练脚本里"先 shuffle 再切"的语义一致，但**不依赖调用方**，
    避免出现"扫参每个配置用不同留出集，结果不可比"的情况。
    """
    import random

    n = len(items)
    h = max(0, min(int(holdout), n))
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    hold = set(idx[:h])
    train = [x for i, x in enumerate(items) if i not in hold]
    evals = [x for i, x in enumerate(items) if i in hold]
    return train, evals
