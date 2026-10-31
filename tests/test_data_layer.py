# -*- coding: utf-8 -*-
"""数据层：多种数据格式规范化 + 流式读取 + 确定性划分。

别人的数据长什么样都有：`{"prompt","completion"}`、`{"text"}`、
ShareGPT 的 `{"messages":[…]}`、Alpaca 的 `{"instruction","output"}`，
还有"大到不能一次读进内存"的 jsonl。这一层负责把它们变成训练循环吃的形状。

关键设计：用**鸭子类型**接 HF datasets（不 import `datasets`），
所以 CI 里没装也能测；错误信息必须写清「第几条 / 什么键 / 怎么修」。
"""
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

from tgai_go.train import data as D


def test_normalize_native_and_text():
    out = D.normalize([{"prompt": "问", "completion": "答"},
                       {"text": "一整段无监督文本"}])
    assert out[0] == {"prompt": "问", "completion": "答"}
    assert out[1] == {"prompt": "", "completion": "一整段无监督文本"}


def test_normalize_sharegpt_messages_takes_last_assistant():
    items = [{"messages": [
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "再加 1 呢"},
        {"role": "assistant", "content": "3"},
    ]}]
    out = D.normalize(items)[0]
    assert out["completion"] == "3", "应该取**最后一条** assistant 当答案"
    assert "1+1=?" in out["prompt"] and "再加 1 呢" in out["prompt"]


def test_normalize_alpaca_style():
    out = D.normalize([{"instruction": "翻成英文：你好", "output": "hello"}])
    assert out[0]["completion"] == "hello"


def test_normalize_rejects_unknown_shape_with_keys_in_message():
    with pytest.raises(D.DataError) as ei:
        D.normalize([{"foo": 1}])
    msg = str(ei.value)
    assert "第 0 条" in msg and "修法" in msg


def test_normalize_rejects_empty_completion():
    with pytest.raises(D.DataError) as ei:
        D.normalize([{"prompt": "问", "completion": ""}])
    assert "空答案" in str(ei.value)


def test_messages_without_assistant_is_rejected():
    with pytest.raises(D.DataError) as ei:
        D.normalize([{"messages": [{"role": "user", "content": "hi"}]}])
    assert "assistant" in str(ei.value) and "修法" in str(ei.value)


def test_read_jsonl_streams_and_reports_bad_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"prompt":"a","completion":"b"}\n\n{"text":"c"}\n', encoding="utf-8")
    rows = D.read_jsonl(str(p))
    assert len(rows) == 2, "空行应被跳过"

    p2 = tmp_path / "bad.jsonl"
    p2.write_text('{"a":1}\n{bad json}\n', encoding="utf-8")
    with pytest.raises(D.DataError) as ei:
        D.read_jsonl(str(p2))
    msg = str(ei.value)
    assert "第 2 行" in msg and "修法" in msg


def test_read_jsonl_limit(tmp_path):
    p = tmp_path / "big.jsonl"
    p.write_text("".join('{"text":"t%d"}\n' % i for i in range(100)), encoding="utf-8")
    assert len(D.read_jsonl(str(p), limit=7)) == 7


def test_from_hf_duck_typed_dataset():
    """不装 datasets 也能接：只要可迭代。"""

    class FakeDataset:
        column_names = ["prompt", "completion"]

        def __iter__(self):
            yield {"prompt": "p1", "completion": "c1"}
            yield {"prompt": "p2", "completion": "c2"}

    out = D.from_hf(FakeDataset())
    assert [x["completion"] for x in out] == ["c1", "c2"]


def test_from_hf_rejects_non_iterable():
    with pytest.raises(D.DataError) as ei:
        D.from_hf(42)
    assert "不可迭代" in str(ei.value) and "修法" in str(ei.value)


def test_train_eval_split_is_deterministic_and_disjoint():
    items = [{"prompt": str(i), "completion": str(i)} for i in range(20)]
    a_tr, a_ev = D.train_eval_split(items, holdout=5, seed=7)
    b_tr, b_ev = D.train_eval_split(items, holdout=5, seed=7)
    assert [x["prompt"] for x in a_ev] == [x["prompt"] for x in b_ev], "同 seed 必须可复现"
    assert len(a_ev) == 5 and len(a_tr) == 15
    assert not ({id(x) for x in a_tr} & {id(x) for x in a_ev}), "训练集与留出集不能重叠"


def test_train_eval_split_holdout_larger_than_data():
    items = [{"prompt": "x", "completion": "y"}]
    tr, ev = D.train_eval_split(items, holdout=10, seed=0)
    assert len(tr) + len(ev) == 1
